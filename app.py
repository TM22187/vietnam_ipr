import argparse
import logging
import queue
import sys
import threading
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from lpr_pipeline import LicensePlateRecognizer, PLATE_CLASS_ID, find_best_model
from run_webcam import ThreadedCamera, load_roi_from_yaml


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("lpr.app")


class CameraWorker(threading.Thread):
    SNAPSHOT_WIDTH_THRESHOLD = 100
    OCR_CACHE_TTL = 60.0
    COOLDOWN_SECONDS = 10.0
    CACHE_CLEANUP_INTERVAL = 100

    def __init__(self, model_path, camera_index, confidence, use_gpu, use_roi, output_queue):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.camera_index = camera_index
        self.confidence = confidence
        self.use_gpu = use_gpu
        self.use_roi = use_roi
        self.output_queue = output_queue
        self.stop_event = threading.Event()

    def _offer(self, item):
        try:
            self.output_queue.put_nowait(item)
            return
        except queue.Full:
            pass

        try:
            self.output_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self.output_queue.put_nowait(item)
        except queue.Full:
            pass

    def stop(self):
        self.stop_event.set()

    def _put_status(self, message):
        self._offer({"type": "status", "message": message})

    def _put_error(self, message):
        self._offer({"type": "error", "message": message})

    def _put_frame(self, frame, fps, recent_plates):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._offer({
            "type": "frame",
            "image": rgb,
            "fps": fps,
            "recent": list(recent_plates),
        })

    def _cleanup_cache(self, cache, cooldown, now):
        cache = {
            key: value for key, value in cache.items()
            if now - value.get("timestamp", 0) < self.OCR_CACHE_TTL
        }
        cooldown = {
            key: value for key, value in cooldown.items()
            if now - value < self.COOLDOWN_SECONDS
        }
        return cache, cooldown

    def _draw_roi(self, frame, recognizer):
        h, w = frame.shape[:2]
        roi_px = recognizer.get_roi_pixels(w, h)
        if not roi_px:
            return frame

        rx1, ry1, rx2, ry2 = roi_px
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, ry1), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, ry2), (w, h), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, ry1), (rx1, ry2), (0, 0, 0), -1)
        cv2.rectangle(overlay, (rx2, ry1), (w, ry2), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 1)
        cv2.putText(frame, "ROI", (rx1 + 5, ry1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return frame

    def run(self):
        if not self.model_path or not Path(self.model_path).exists():
            self._put_error("Khong tim thay model weights.")
            return

        cap = None
        try:
            roi = load_roi_from_yaml() if self.use_roi else None
            self._put_status("Dang tai model...")
            recognizer = LicensePlateRecognizer(
                self.model_path,
                confidence_threshold=self.confidence,
                use_gpu=self.use_gpu,
                roi=roi,
            )

            device = "cuda" if self.use_gpu else "cpu"
            recognizer.yolo.to(device)
            cap = ThreadedCamera(self.camera_index)
            time.sleep(0.8)

            self._put_status("Dang chay camera")
            ocr_cache = {}
            plate_cooldown = {}
            recent_plates = deque(maxlen=5)
            fps_times = deque(maxlen=30)
            frame_count = 0
            last_ui_update = 0.0
            font = cv2.FONT_HERSHEY_SIMPLEX

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.03)
                    continue

                now = time.time()
                frame_count += 1
                fps_times.append(now)
                fps = 0.0
                if len(fps_times) > 1:
                    fps = (len(fps_times) - 1) / max(0.001, fps_times[-1] - fps_times[0])

                h, w = frame.shape[:2]
                output = self._draw_roi(frame.copy(), recognizer)

                if frame_count % self.CACHE_CLEANUP_INTERVAL == 0:
                    ocr_cache, plate_cooldown = self._cleanup_cache(ocr_cache, plate_cooldown, now)

                results = recognizer.yolo.track(
                    frame,
                    conf=recognizer.conf_threshold,
                    persist=True,
                    verbose=False,
                    imgsz=640,
                    device=device,
                    half=self.use_gpu,
                )

                if results and results[0].boxes is not None and results[0].boxes.id is not None:
                    for box in results[0].boxes:
                        if int(box.cls[0]) != PLATE_CLASS_ID:
                            continue

                        track_id = int(box.id[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        if not recognizer._is_inside_roi(x1, y1, x2, y2, w, h):
                            continue

                        cached = ocr_cache.get(track_id, {})
                        text = cached.get("text", "")
                        ocr_conf = cached.get("ocr_conf", 0.0)
                        is_valid = cached.get("is_valid", False)
                        box_width = x2 - x1

                        if not is_valid and box_width > self.SNAPSHOT_WIDTH_THRESHOLD:
                            crop = frame[
                                max(0, y1 - 20): min(h, y2 + 20),
                                max(0, x1 - 20): min(w, x2 + 20),
                            ]
                            if crop.size > 0:
                                cleaned, conf, raw, valid = recognizer.ocr_plate_crop(crop)
                                if valid or conf > 0.4:
                                    text, ocr_conf, is_valid = cleaned, conf, valid
                                    ocr_cache[track_id] = {
                                        "text": text,
                                        "ocr_conf": ocr_conf,
                                        "is_valid": is_valid,
                                        "timestamp": now,
                                    }

                                    if text and (valid or (len(text) >= 5 and ocr_conf > 0.6)):
                                        last_seen = plate_cooldown.get(text, 0)
                                        if now - last_seen > self.COOLDOWN_SECONDS:
                                            recent_plates.appendleft(f"{text} ({ocr_conf:.0%})")
                                            plate_cooldown[text] = now
                                            ocr_cache[track_id]["is_valid"] = True
                                            is_valid = True

                        color = (0, 255, 0) if is_valid else (0, 215, 255)
                        label = f"{text} ({ocr_conf:.0%})" if text else "Tien lai gan hon..."
                        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(output, label, (x1, max(20, y1 - 10)),
                                    font, 0.7, color, 2, cv2.LINE_AA)

                cv2.putText(output, f"FPS: {fps:.1f}", (10, 30), font, 0.7, (0, 255, 255), 2)

                if now - last_ui_update >= 1 / 30:
                    self._put_frame(output, fps, recent_plates)
                    last_ui_update = now

        except Exception as exc:
            logger.exception("App worker failed")
            self._put_error(str(exc))
        finally:
            if cap:
                cap.release()
            self._put_status("Da dung")


class LprApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vietnam LPR")
        self.geometry("1120x760")
        self.minsize(920, 620)

        self.output_queue = queue.Queue(maxsize=4)
        self.worker = None
        self.current_image = None

        self.model_var = tk.StringVar(value=find_best_model() or str(ROOT / "weights" / "best_vietnam_lpr.pt"))
        self.camera_var = tk.IntVar(value=0)
        self.conf_var = tk.DoubleVar(value=0.25)
        self.gpu_var = tk.BooleanVar(value=False)
        self.roi_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="San sang")
        self.fps_var = tk.StringVar(value="FPS: 0.0")

        self._build_ui()
        self.after(30, self._poll_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=0)
        root.rowconfigure(0, weight=1)

        self.video_label = ttk.Label(root, anchor="center")
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        panel = ttk.Frame(root, width=300)
        panel.grid(row=0, column=1, sticky="ns")
        panel.columnconfigure(0, weight=1)

        ttk.Label(panel, text="Model weights").grid(row=0, column=0, sticky="w")
        ttk.Entry(panel, textvariable=self.model_var, width=42).grid(row=1, column=0, sticky="ew", pady=(4, 10))

        ttk.Label(panel, text="Camera").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(panel, from_=0, to=9, textvariable=self.camera_var, width=8).grid(row=3, column=0, sticky="w", pady=(4, 10))

        ttk.Label(panel, text="Detection confidence").grid(row=4, column=0, sticky="w")
        ttk.Scale(panel, from_=0.05, to=0.95, variable=self.conf_var).grid(row=5, column=0, sticky="ew", pady=(4, 10))

        ttk.Checkbutton(panel, text="Dung ROI neu co config/roi.yaml", variable=self.roi_var).grid(row=6, column=0, sticky="w")
        ttk.Checkbutton(panel, text="Dung GPU cho YOLO", variable=self.gpu_var).grid(row=7, column=0, sticky="w", pady=(0, 12))

        controls = ttk.Frame(panel)
        controls.grid(row=8, column=0, sticky="ew", pady=(4, 16))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        self.start_button = ttk.Button(controls, text="Start", command=self.start_camera)
        self.stop_button = ttk.Button(controls, text="Stop", command=self.stop_camera, state="disabled")
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        ttk.Label(panel, textvariable=self.status_var).grid(row=9, column=0, sticky="w", pady=(0, 4))
        ttk.Label(panel, textvariable=self.fps_var).grid(row=10, column=0, sticky="w", pady=(0, 16))

        ttk.Label(panel, text="5 bien so gan nhat").grid(row=11, column=0, sticky="w")
        self.history = tk.Listbox(panel, height=8)
        self.history.grid(row=12, column=0, sticky="ew", pady=(4, 0))

    def start_camera(self):
        if self.worker and self.worker.is_alive():
            return

        self.worker = CameraWorker(
            self.model_var.get(),
            self.camera_var.get(),
            self.conf_var.get(),
            self.gpu_var.get(),
            self.roi_var.get(),
            self.output_queue,
        )
        self.worker.start()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("Dang khoi dong...")

    def stop_camera(self):
        if self.worker:
            self.worker.stop()
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.status_var.set("Dang dung...")

    def _poll_queue(self):
        try:
            while True:
                item = self.output_queue.get_nowait()
                if item["type"] == "frame":
                    self._show_frame(item["image"])
                    self.fps_var.set(f"FPS: {item['fps']:.1f}")
                    self._set_recent(item["recent"])
                elif item["type"] == "status":
                    self.status_var.set(item["message"])
                elif item["type"] == "error":
                    self.stop_camera()
                    messagebox.showerror("Vietnam LPR", item["message"])
        except queue.Empty:
            pass

        self.after(30, self._poll_queue)

    def _show_frame(self, rgb):
        width = max(1, self.video_label.winfo_width())
        height = max(1, self.video_label.winfo_height())
        image = Image.fromarray(rgb)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.current_image = ImageTk.PhotoImage(image)
        self.video_label.configure(image=self.current_image)

    def _set_recent(self, items):
        self.history.delete(0, tk.END)
        for item in items:
            self.history.insert(tk.END, item)

    def _on_close(self):
        self.stop_camera()
        self.destroy()


def parse_args():
    parser = argparse.ArgumentParser(description="Vietnam LPR desktop app")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--no-roi", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    app = LprApp()
    if args.model:
        app.model_var.set(args.model)
    app.camera_var.set(args.camera)
    app.conf_var.set(args.conf)
    app.gpu_var.set(args.gpu)
    app.roi_var.set(not args.no_roi)
    app.mainloop()


if __name__ == "__main__":
    main()
