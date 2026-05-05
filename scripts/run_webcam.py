"""
Vietnam License Plate Recognition - Real-time Webcam Demo
==========================================================
Architecture:
    Main thread  → YOLO tracking + PlateTracker voting on OCR cache (stable labels)
    OCR thread   → OCR only when needed; smoothing reduces jitter on moving plates

Usage (from repo root):
    python scripts/run_webcam.py
    python scripts/run_webcam.py --camera 1 --model weights/best_vietnam_lpr.pt --conf 0.25

Hotkeys:
    q  - Quit
    s  - Save frame → captures/
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import time
import queue
import threading
import argparse
import cv2

from lpr_pipeline import (
    LicensePlateRecognizer,
    TrackedPlateCache,
    PlateTracker,
    prepare_plate_recognitions,
    find_best_model,
    ocr_worker,
    MIN_BLUR_SCORE,
    OCR_QUEUE_MAXSIZE,
    SMOOTH_WINDOW,
    SMOOTH_MIN_VOTES,
    SMOOTH_STALE_FRAMES,
)

CAPTURE_DIR = ROOT / "captures"


def run_webcam_demo(recognizer, camera_index=0, show_fps=True):
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_index}")
        print("        Try --camera 1 if you have multiple webcams.")
        return

    print("[OK] Webcam is running!")
    print("     Press 'q' to quit | 's' to save frame")

    fps_counter, fps_start = 0, time.time()
    current_fps = 0.0
    frame_count = 0

    plate_cache = TrackedPlateCache(expire_seconds=5.0)
    plate_tracker = PlateTracker(
        window=SMOOTH_WINDOW,
        min_votes=SMOOTH_MIN_VOTES,
        stale_limit=SMOOTH_STALE_FRAMES,
    )

    ocr_q = queue.Queue(maxsize=OCR_QUEUE_MAXSIZE)
    stop_event = threading.Event()
    worker = threading.Thread(
        target=ocr_worker,
        args=(recognizer, plate_cache, ocr_q, stop_event),
        daemon=True,
    )
    worker.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read frame from camera.")
            break

        frame_count += 1

        tracking = recognizer.track_frame(frame, imgsz=480)

        for plate in tracking["plates"]:
            tid = plate["track_id"]
            blur = plate["blur_score"]
            plate_cache.mark_seen(tid)

            if blur < MIN_BLUR_SCORE:
                continue

            if plate_cache.needs_ocr(tid, blur):
                try:
                    ocr_q.put_nowait((tid, plate["crop"].copy(), blur))
                except queue.Full:
                    pass

        if frame_count % 90 == 0:
            plate_cache.cleanup()

        recs = prepare_plate_recognitions(tracking, plate_cache)
        smoothed = plate_tracker.update(recs)
        output = recognizer.draw_tracked_results(frame, tracking, smoothed)

        fps_counter += 1
        if fps_counter >= 30:
            elapsed = time.time() - fps_start
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()

        if show_fps:
            n_vehicles = len(tracking["vehicles"])
            n_plates = len(tracking["plates"])
            cv2.putText(output, f"FPS: {current_fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)
            cv2.putText(output, f"Vehicles: {n_vehicles}  Plates: {n_plates}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)
            y_pos = 90
            for plate, row in zip(tracking["plates"], smoothed):
                text = (row.get("text") or "").strip()
                if text:
                    conf = float(row.get("ocr_conf") or 0.0)
                    good = "OK" if row.get("is_valid") else "?"
                    label = f"#{plate['track_id']}: {text} ({conf:.0%}) [{good}]"
                    color = (0, 255, 100) if row.get("is_valid") else (
                        0, 200, 255)
                else:
                    label = f"#{plate['track_id']}: ..."
                    color = (180, 180, 180)
                cv2.putText(output, label, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                y_pos += 25

        cv2.imshow("Vietnam LPR - Webcam  (Q: Quit, S: Save)", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quit.")
            break
        elif key == ord("s"):
            CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
            save_path = CAPTURE_DIR / f"capture_{frame_count}.jpg"
            cv2.imwrite(str(save_path), output)
            print(f"Saved: {save_path}")

    stop_event.set()
    ocr_q.put(None)
    worker.join(timeout=2)
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Real-time Vietnamese LPR from webcam"
    )
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--model",  default=None,
                        help="Path to .pt model file")
    parser.add_argument("--conf",   type=float, default=0.25,
                        help="Detection confidence threshold")
    parser.add_argument("--gpu",    action="store_true",
                        help="Use GPU for PaddleOCR (default: CPU)")
    args = parser.parse_args()

    model_path = args.model or find_best_model()
    if not model_path or not os.path.exists(model_path):
        print(
            "[ERROR] Model file not found. Place weights under weights/ "
            "or pass --model path/to/best.pt")
        return

    recognizer = LicensePlateRecognizer(
        model_path, confidence_threshold=args.conf, use_gpu=args.gpu)
    run_webcam_demo(recognizer, camera_index=args.camera)


if __name__ == "__main__":
    main()
