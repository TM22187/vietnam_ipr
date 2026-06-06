"""
Nhận Dạng Biển Số Xe Việt Nam - Phân Tích Video (Chế Độ Cổng)
==============================================================
Kiến trúc:
    Luồng chính → YOLO detection → PaddleOCR (đồng bộ) → Kiểm tra cooldown.
    Không tracking, không hàng đợi bất đồng bộ. Phù hợp cho xe chạy chậm hoặc đứng yên tại cổng.

Cách dùng (từ thư mục gốc dự án):
    python scripts/run_video.py --video path/to/video.mp4
    python scripts/run_video.py --video video.mp4 --output result.mp4 --max-frames 500
    python scripts/run_video.py --video video.mp4 --conf 0.4

Phím tắt (trong cửa sổ xem):
    q  - Thoát sớm
    s  - Lưu frame hiện tại dạng JPEG → thư mục captures/
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import time
import logging
import argparse
import cv2

from lpr_pipeline import (
    LicensePlateRecognizer,
    find_best_model,
)

logger = logging.getLogger("lpr.video")
CAPTURE_DIR = ROOT / "captures"


def run_video(recognizer, video_path, output_path=None,
              max_frames=None, preview=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Không thể mở video: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = max_frames if max_frames else total

    logger.info(f"Video: {w}x{h}  {fps} FPS  tổng {total} frame")
    logger.info(f"Xử lý tối đa {limit} frame...")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_count = 0
    fps_counter, fps_start = 0, time.time()
    current_fps = 0.0

    cooldown_cache = {}
    COOLDOWN_SECONDS = 5.0

    while frame_count < limit:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        now = time.time()

        # Dọn cooldown đã hết hạn
        cooldown_cache = {k: v for k, v in cooldown_cache.items() if now - v < COOLDOWN_SECONDS}

        # Nhận dạng đồng bộ
        recognitions = recognizer.recognize(frame)
        output_frame = recognizer.draw_results(frame, recognitions)

        for rec in recognitions:
            if rec["is_valid"] and rec["ocr_conf"] > 0.8:
                text = rec["text"]
                if text not in cooldown_cache:
                    logger.info(f">>> [MO CONG] Phat hien bien so hop le: {text} (Do tu tin: {rec['ocr_conf']:.0%})")
                    cooldown_cache[text] = now

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
            cv2.imshow("Nhan Dang Bien So - Phan Tich Video  (Q: Thoat, S: Luu)", output_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Người dùng dừng sớm.")
                break
            elif key == ord("s"):
                CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
                save_path = CAPTURE_DIR / f"capture_{frame_count}.jpg"
                cv2.imwrite(str(save_path), output_frame)
                logger.info(f"Đã lưu: {save_path}")

        if frame_count % 50 == 0:
            logger.info(f"  Đã xử lý {frame_count}/{limit} frame ...")

    cap.release()
    if writer:
        writer.release()
        logger.info(f"Video đã chú thích lưu tại: {output_path}")
    if preview:
        cv2.destroyAllWindows()

    logger.info(f"Hoàn tất. Tổng số frame đã xử lý: {frame_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Phân tích video nhận dạng biển số xe Việt Nam (Chế Độ Cổng)"
    )
    parser.add_argument("--video",      required=True,
                        help="Đường dẫn file video đầu vào")
    parser.add_argument("--output",     default=None,
                        help="Đường dẫn lưu video đã chú thích (tùy chọn)")
    parser.add_argument("--model",      default=None,
                        help="Đường dẫn file model .pt")
    parser.add_argument("--conf",       type=float, default=0.5,
                        help="Ngưỡng confidence detection")
    parser.add_argument("--gpu",        action="store_true",
                        help="Dùng GPU cho YOLO (PaddleOCR luôn CPU)")
    parser.add_argument("--max-frames", type=int,   default=None,
                        help="Số frame tối đa cần xử lý")
    parser.add_argument("--no-preview", action="store_true",
                        help="Tắt cửa sổ xem trực tiếp")
    args = parser.parse_args()

    model_path = args.model or find_best_model()
    if not model_path or not os.path.exists(model_path):
        print(
            "[LỖI] Không tìm thấy file model. Đặt weights vào thư mục weights/ "
            "hoặc truyền --model path/to/best.pt")
        return

    # ── Cấu hình logging ──
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

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
