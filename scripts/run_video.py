"""
Vietnam License Plate Recognition - Test with Video File
=========================================================
Architecture:
    Main thread  → YOLO tracking on EVERY frame  (ByteTrack + local yaml)
                 → merges OCR cache + PlateTracker temporal voting (stable text)
    OCR thread   → reads plate text only for NEW / SHARPER crops
                 → caches per track ID; smoothing hides jitter when reads disagree

Usage (from repo root):
    python scripts/run_video.py --video path/to/video.mp4
    python scripts/run_video.py --video video.mp4 --output result.mp4 --max-frames 500
    python scripts/run_video.py --video video.mp4 --conf 0.4

Hotkeys (in the preview window):
    q  - Quit early
    s  - Save current frame as JPEG → captures/
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


def run_video(recognizer, video_path, output_path=None,
              max_frames=None, preview=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = max_frames if max_frames else total

    print(f"Video : {w}x{h}  {fps} FPS  {total} frames total")
    print(f"Processing up to {limit} frames...")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

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

    frame_count = 0
    fps_counter, fps_start = 0, time.time()
    current_fps = 0.0

    while frame_count < limit:
        ret, frame = cap.read()
        if not ret:
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
        output_frame = recognizer.draw_tracked_results(
            frame, tracking, smoothed)

        fps_counter += 1
        if fps_counter >= 30:
            elapsed = time.time() - fps_start
            current_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()

        cv2.putText(output_frame, f"FPS: {current_fps:.1f}  Frame: {frame_count}/{limit}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        if writer:
            writer.write(output_frame)

        if preview:
            cv2.imshow("Vietnam LPR - Video  (Q: Quit, S: Save)", output_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Stopped early by user.")
                break
            elif key == ord("s"):
                CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
                save_path = CAPTURE_DIR / f"capture_{frame_count}.jpg"
                cv2.imwrite(str(save_path), output_frame)
                print(f"Saved: {save_path}")

        if frame_count % 50 == 0:
            print(f"  Processed {frame_count}/{limit} frames ...")

    stop_event.set()
    ocr_q.put(None)
    worker.join(timeout=2)
    cap.release()
    if writer:
        writer.release()
        print(f"[DONE] Annotated video saved to: {output_path}")
    if preview:
        cv2.destroyAllWindows()

    print(f"Finished. Total frames processed: {frame_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Test Vietnamese LPR on a video file"
    )
    parser.add_argument("--video",      required=True,
                        help="Path to input video file")
    parser.add_argument("--output",     default=None,
                        help="Path to save annotated video (optional)")
    parser.add_argument("--model",      default=None,
                        help="Path to .pt model file")
    parser.add_argument("--conf",       type=float, default=0.5,
                        help="Detection confidence threshold")
    parser.add_argument("--gpu",        action="store_true",
                        help="Use GPU for PaddleOCR (default: CPU)")
    parser.add_argument("--max-frames", type=int,   default=None,
                        help="Maximum number of frames to process")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable live preview window")
    args = parser.parse_args()

    model_path = args.model or find_best_model()
    if not model_path or not os.path.exists(model_path):
        print(
            "[ERROR] Model file not found. Place weights under weights/ "
            "or pass --model path/to/best.pt")
        return

    recognizer = LicensePlateRecognizer(
        model_path, confidence_threshold=args.conf, use_gpu=args.gpu)
    run_video(
        recognizer,
        video_path=args.video,
        output_path=args.output,
        max_frames=args.max_frames,
        preview=not args.no_preview,
    )


if __name__ == "__main__":
    main()
