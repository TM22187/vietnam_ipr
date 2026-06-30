"""Giao diện desktop tối giản cho Vietnam LPR."""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk

from lpr_pipeline import LicensePlateRecognizer, find_best_model


APP_NAME = "Vietnam LPR"
APP_VERSION = "1.0.0"
BG = "#0f172a"
PANEL = "#172033"
PANEL_ALT = "#111a2c"
TEXT = "#f8fafc"
MUTED = "#94a3b8"
ACCENT = "#22c55e"
ACCENT_HOVER = "#16a34a"


def read_image(path: str) -> np.ndarray | None:
    """Đọc được cả đường dẫn Unicode trên Windows."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


class LatestCamera:
    """Luôn cung cấp frame camera mới nhất để giao diện không bị trễ."""

    def __init__(self, index: int = 0):
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(index, backend)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = cv2.VideoCapture(index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Không thể mở camera {index}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        while self._running:
            ok, frame = self.capture.read()
            if ok:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.03)

    def read(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def release(self) -> None:
        self._running = False
        self.capture.release()


class RecognitionWorker:
    def __init__(self, events: queue.Queue, stop_event: threading.Event):
        self.events = events
        self.stop_event = stop_event
        self.recognizer: LicensePlateRecognizer | None = None

    def _status(self, text: str) -> None:
        self.events.put(("status", text))

    def _load_model(self) -> LicensePlateRecognizer:
        if self.recognizer is None:
            self._status("Đang khởi tạo mô hình (lần đầu có thể mất vài giây)…")
            model = find_best_model()
            if not model:
                raise FileNotFoundError("Thiếu model models/best_vietnam_lpr.onnx")
            self.recognizer = LicensePlateRecognizer(model)
        self.recognizer.reset_stream()
        return self.recognizer

    def run(self, source_type: str, source: str | int) -> None:
        try:
            recognizer = self._load_model()
            if self.stop_event.is_set():
                return
            if source_type == "image":
                self._run_image(recognizer, str(source))
            elif source_type == "video":
                self._run_video(recognizer, str(source))
            else:
                self._run_camera(recognizer, int(source))
        except Exception as exc:
            logging.exception("Lỗi xử lý nguồn")
            self.events.put(("error", str(exc)))
        finally:
            self.events.put(("done", None))

    def _run_image(self, recognizer: LicensePlateRecognizer, path: str) -> None:
        frame = read_image(path)
        if frame is None:
            raise RuntimeError("Không đọc được file ảnh đã chọn")
        self._status("Đang nhận dạng ảnh…")
        results = recognizer.recognize(frame)
        output = recognizer.draw_results(frame, results)
        self.events.put(("frame", output))
        for result in results:
            if result.get("text"):
                self.events.put(("plate", result))
        if results:
            self._status(f"Hoàn tất · tìm thấy {len(results)} biển số")
        else:
            self._status("Hoàn tất · không tìm thấy biển số")

    def _run_video(self, recognizer: LicensePlateRecognizer, path: str) -> None:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise RuntimeError("Không mở được file video đã chọn")
        total = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        # Giới hạn inference khoảng 12 frame/s; bỏ frame làm UI mượt hơn trên CPU.
        stride = max(1, round(fps / 12.0))
        frame_index = 0
        self._status("Đang phân tích video…")
        try:
            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1
                if frame_index % stride:
                    continue
                output, _, new_events = recognizer.process_stream_frame(frame)
                self.events.put(("frame", output))
                for event in new_events:
                    self.events.put(("plate", event))
                if total and frame_index % max(stride * 15, 1) == 0:
                    self._status(f"Đang phân tích video · {frame_index * 100 // total}%")
        finally:
            capture.release()
        if not self.stop_event.is_set():
            self._status("Đã phân tích xong video")

    def _run_camera(self, recognizer: LicensePlateRecognizer, index: int) -> None:
        camera = LatestCamera(index)
        self._status("Camera đang hoạt động")
        try:
            while not self.stop_event.is_set():
                frame = camera.read()
                if frame is None:
                    time.sleep(0.02)
                    continue
                output, _, new_events = recognizer.process_stream_frame(frame)
                self.events.put(("frame", output))
                for event in new_events:
                    self.events.put(("plate", event))
        finally:
            camera.release()


class VietnamLPRApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"{APP_NAME} · Nhận dạng biển số")
        self.root.geometry("1180x720")
        self.root.minsize(920, 600)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        self.events: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = RecognitionWorker(self.events, self.stop_event)
        self.worker_thread: threading.Thread | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.last_frame: np.ndarray | None = None

        self._configure_styles()
        self._build_ui()
        self.root.after(35, self._poll_events)

    def _configure_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Title.TLabel", background=BG, foreground=TEXT,
                        font=("Segoe UI Semibold", 20))
        style.configure("Subtitle.TLabel", background=BG, foreground=MUTED,
                        font=("Segoe UI", 10))
        style.configure("PanelTitle.TLabel", background=PANEL, foreground=TEXT,
                        font=("Segoe UI Semibold", 12))
        style.configure("Status.TLabel", background=PANEL_ALT, foreground=MUTED,
                        font=("Segoe UI", 10), padding=(12, 9))
        style.configure("Action.TButton", font=("Segoe UI Semibold", 10), padding=(14, 9),
                        background=ACCENT, foreground="#052e16", borderwidth=0)
        style.map("Action.TButton", background=[("active", ACCENT_HOVER), ("disabled", "#334155")],
                  foreground=[("disabled", "#94a3b8")])
        style.configure("Secondary.TButton", font=("Segoe UI", 10), padding=(13, 9),
                        background="#263449", foreground=TEXT, borderwidth=0)
        style.map("Secondary.TButton", background=[("active", "#334155")])
        style.configure("Danger.TButton", font=("Segoe UI Semibold", 10), padding=(13, 9),
                        background="#7f1d1d", foreground="#fecaca", borderwidth=0)
        style.map("Danger.TButton", background=[("active", "#991b1b")])
        style.configure("Treeview", background=PANEL_ALT, fieldbackground=PANEL_ALT,
                        foreground=TEXT, rowheight=34, borderwidth=0,
                        font=("Segoe UI", 10))
        style.configure("Treeview.Heading", background="#263449", foreground=MUTED,
                        font=("Segoe UI Semibold", 9), borderwidth=0)
        style.map("Treeview", background=[("selected", "#1e3a5f")])

    def _build_ui(self) -> None:
        header = ttk.Frame(self.root, padding=(24, 18, 24, 12))
        header.pack(fill="x")
        title_block = ttk.Frame(header)
        title_block.pack(side="left")
        ttk.Label(title_block, text="Vietnam LPR", style="Title.TLabel").pack(anchor="w")
        ttk.Label(title_block, text="Nhận dạng biển số Việt Nam · xử lý offline",
                  style="Subtitle.TLabel").pack(anchor="w", pady=(2, 0))

        controls = ttk.Frame(header)
        controls.pack(side="right")
        self.image_button = ttk.Button(controls, text="Mở ảnh", style="Secondary.TButton",
                                       command=self._choose_image)
        self.image_button.pack(side="left", padx=4)
        self.video_button = ttk.Button(controls, text="Mở video", style="Secondary.TButton",
                                       command=self._choose_video)
        self.video_button.pack(side="left", padx=4)
        self.camera_button = ttk.Button(controls, text="Bật camera", style="Action.TButton",
                                        command=lambda: self._start("camera", 0, "Camera 0"))
        self.camera_button.pack(side="left", padx=4)
        self.stop_button = ttk.Button(controls, text="Dừng", style="Danger.TButton",
                                      command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))

        content = ttk.Frame(self.root, padding=(24, 0, 24, 18))
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=7)
        content.columnconfigure(1, weight=3)
        content.rowconfigure(0, weight=1)

        preview_panel = ttk.Frame(content, style="Panel.TFrame", padding=12)
        preview_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        preview_panel.rowconfigure(1, weight=1)
        preview_panel.columnconfigure(0, weight=1)
        self.source_label = ttk.Label(preview_panel, text="Chưa chọn nguồn", style="PanelTitle.TLabel")
        self.source_label.grid(row=0, column=0, sticky="w", pady=(0, 9))
        self.preview = tk.Label(preview_panel, text="Chọn ảnh, video hoặc bật camera để bắt đầu",
                                bg="#090f1d", fg=MUTED, font=("Segoe UI", 11),
                                compound="center")
        self.preview.grid(row=1, column=0, sticky="nsew")
        self.preview.bind("<Configure>", lambda _event: self._render_frame())

        history_panel = ttk.Frame(content, style="Panel.TFrame", padding=12)
        history_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        history_panel.rowconfigure(1, weight=1)
        history_panel.columnconfigure(0, weight=1)
        history_header = ttk.Frame(history_panel, style="Panel.TFrame")
        history_header.grid(row=0, column=0, sticky="ew", pady=(0, 9))
        ttk.Label(history_header, text="Kết quả gần đây", style="PanelTitle.TLabel").pack(side="left")
        ttk.Button(history_header, text="Xóa", style="Secondary.TButton",
                   command=self._clear_history).pack(side="right")

        self.history = ttk.Treeview(history_panel, columns=("plate", "confidence", "time"),
                                    show="headings", selectmode="browse")
        self.history.heading("plate", text="BIỂN SỐ")
        self.history.heading("confidence", text="TIN CẬY")
        self.history.heading("time", text="THỜI GIAN")
        self.history.column("plate", width=130, minwidth=95, anchor="w")
        self.history.column("confidence", width=78, minwidth=65, anchor="center")
        self.history.column("time", width=78, minwidth=65, anchor="center")
        self.history.grid(row=1, column=0, sticky="nsew")

        self.status = ttk.Label(self.root, text="Sẵn sàng", style="Status.TLabel", anchor="w")
        self.status.pack(fill="x", side="bottom")

    def _choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Chọn ảnh", filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Tất cả", "*.*")],
        )
        if path:
            self._start("image", path, Path(path).name)

    def _choose_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Chọn video", filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("Tất cả", "*.*")],
        )
        if path:
            self._start("video", path, Path(path).name)

    def _start(self, source_type: str, source: str | int, label: str) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            return
        self.stop_event.clear()
        self.source_label.configure(text=label)
        self._set_running(True)
        self.worker_thread = threading.Thread(
            target=self.worker.run, args=(source_type, source), daemon=True,
        )
        self.worker_thread.start()

    def _stop(self) -> None:
        self.stop_event.set()
        self.status.configure(text="Đang dừng…")
        self.stop_button.configure(state="disabled")

    def _set_running(self, running: bool) -> None:
        source_state = "disabled" if running else "normal"
        self.image_button.configure(state=source_state)
        self.video_button.configure(state=source_state)
        self.camera_button.configure(state=source_state)
        self.stop_button.configure(state="normal" if running else "disabled")

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "frame":
                    self.last_frame = payload
                    self._render_frame()
                elif kind == "plate":
                    self._add_plate(payload)
                elif kind == "status":
                    self.status.configure(text=payload)
                elif kind == "error":
                    self.status.configure(text=f"Lỗi: {payload}")
                    messagebox.showerror(APP_NAME, payload)
                elif kind == "done":
                    self._set_running(False)
        except queue.Empty:
            pass
        self.root.after(35, self._poll_events)

    def _render_frame(self) -> None:
        if self.last_frame is None:
            return
        width = max(64, self.preview.winfo_width() - 4)
        height = max(64, self.preview.winfo_height() - 4)
        rgb = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image = ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.preview_photo, text="")

    def _add_plate(self, result: dict) -> None:
        text = result.get("text", "")
        if not text:
            return
        confidence = float(result.get("ocr_conf", 0.0))
        self.history.insert("", 0, values=(text, f"{confidence:.0%}", datetime.now().strftime("%H:%M:%S")))
        items = self.history.get_children()
        for item in items[50:]:
            self.history.delete(item)

    def _clear_history(self) -> None:
        for item in self.history.get_children():
            self.history.delete(item)

    def _close(self) -> None:
        self.stop_event.set()
        self.root.destroy()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if "--smoke-test" in sys.argv:
        # Được dùng sau khi đóng gói: xác nhận Tk, ONNX Runtime, OCR và model
        # đều có thể khởi tạo trong chính file .exe phân phối.
        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        root.destroy()
        LicensePlateRecognizer()
        return
    root = tk.Tk()
    VietnamLPRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
