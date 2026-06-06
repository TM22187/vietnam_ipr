"""
Nhận Dạng Biển Số Xe Việt Nam - Điểm Khởi Chạy Demo
====================================================
Cách dùng:
    python demo.py --mode webcam
    python demo.py --mode webcam  --camera 1 --conf 0.25
    python demo.py --mode video   --video clip.mp4
    python demo.py --mode video   --video clip.mp4 --output result.mp4
    python demo.py --mode image   --image photo.jpg
"""

import argparse
import os
import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lpr_pipeline import LicensePlateRecognizer, find_best_model


def _load_recognizer(model_path, conf, gpu, roi=None):
    path = model_path or find_best_model()
    if not path or not os.path.exists(path):
        print("[LỖI] Không tìm thấy model. Hãy đặt weights vào thư mục weights/ hoặc truyền --model.")
        sys.exit(1)
    return LicensePlateRecognizer(path, confidence_threshold=conf, use_gpu=gpu, roi=roi)


def main():
    parser = argparse.ArgumentParser(
        description="Demo Nhận Dạng Biển Số Xe Việt Nam",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
ví dụ:
  python demo.py --mode webcam
  python demo.py --mode webcam  --camera 1 --conf 0.25 --gpu
  python demo.py --mode video   --video clip.mp4
  python demo.py --mode video   --video clip.mp4 --output result.mp4 --no-preview
  python demo.py --mode image   --image photo.jpg --conf 0.4
        """,
    )

    parser.add_argument("--mode", choices=["webcam", "video", "image"], required=True,
                        help="Nguồn đầu vào: webcam | video | image")
    parser.add_argument("--model", default=None,
                        help="Đường dẫn file weights .pt (tự tìm nếu bỏ qua)")
    parser.add_argument("--conf",  type=float, default=0.25,
                        help="Ngưỡng confidence detection (mặc định: 0.25)")
    parser.add_argument("--gpu",   action="store_true",
                        help="Dùng GPU cho YOLO (PaddleOCR luôn chạy CPU)")

    # --- webcam ---
    parser.add_argument("--camera", type=int, default=0,
                        help="[webcam] Chỉ số camera (mặc định: 0)")

    # --- video ---
    parser.add_argument("--video",      default=None,
                        help="[video] Đường dẫn file video đầu vào")
    parser.add_argument("--output",     default=None,
                        help="[video] Đường dẫn lưu video đã chú thích")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="[video] Dừng sau N frame")
    parser.add_argument("--no-preview", action="store_true",
                        help="[video] Tắt cửa sổ xem trực tiếp")

    # --- ảnh tĩnh ---
    parser.add_argument("--image", default=None,
                        help="[image] Đường dẫn ảnh đầu vào")

    args = parser.parse_args()

    # Kiểm tra tham số bắt buộc theo mode
    if args.mode == "video" and not args.video:
        parser.error("--mode video yêu cầu --video <đường_dẫn>")
    if args.mode == "image" and not args.image:
        parser.error("--mode image yêu cầu --image <đường_dẫn>")

    # ── Cấu hình logging ──
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Tải ROI từ config ──
    roi = None
    roi_path = ROOT / "config" / "roi.yaml"
    if roi_path.exists():
        try:
            from run_webcam import load_roi_from_yaml
            roi = load_roi_from_yaml(roi_path)
        except Exception:
            pass  # ROI là tùy chọn, không bắt buộc

    recognizer = _load_recognizer(args.model, args.conf, args.gpu, roi=roi)

    if args.mode == "webcam":
        from run_webcam import run_webcam_demo
        run_webcam_demo(recognizer, camera_index=args.camera)

    elif args.mode == "video":
        from run_video import run_video
        run_video(
            recognizer,
            video_path=args.video,
            output_path=args.output,
            max_frames=args.max_frames,
            preview=not args.no_preview,
        )

    elif args.mode == "image":
        from test_image import test_with_image
        test_with_image(recognizer, args.image)


if __name__ == "__main__":
    main()
