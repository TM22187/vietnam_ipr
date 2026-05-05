"""
Vietnam License Plate Recognition - Demo Entry Point
=====================================================
Usage:
    python demo.py --mode webcam
    python demo.py --mode webcam  --camera 1 --conf 0.25
    python demo.py --mode video   --video clip.mp4
    python demo.py --mode video   --video clip.mp4 --output result.mp4
    python demo.py --mode image   --image photo.jpg
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lpr_pipeline import LicensePlateRecognizer, find_best_model


def _load_recognizer(model_path, conf, gpu):
    path = model_path or find_best_model()
    if not path or not os.path.exists(path):
        print("[ERROR] Model not found. Place weights under weights/ or pass --model.")
        sys.exit(1)
    return LicensePlateRecognizer(path, confidence_threshold=conf, use_gpu=gpu)


def main():
    parser = argparse.ArgumentParser(
        description="Vietnam License Plate Recognition Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python demo.py --mode webcam
  python demo.py --mode webcam  --camera 1 --conf 0.25 --gpu
  python demo.py --mode video   --video clip.mp4
  python demo.py --mode video   --video clip.mp4 --output result.mp4 --no-preview
  python demo.py --mode image   --image photo.jpg --conf 0.4
        """,
    )

    parser.add_argument("--mode", choices=["webcam", "video", "image"], required=True,
                        help="Input source: webcam | video | image")
    parser.add_argument("--model", default=None,
                        help="Path to .pt weights file (auto-detected if omitted)")
    parser.add_argument("--conf",  type=float, default=0.25,
                        help="Detection confidence threshold (default: 0.25)")
    parser.add_argument("--gpu",   action="store_true",
                        help="Use GPU for PaddleOCR (default: CPU)")

    # webcam
    parser.add_argument("--camera", type=int, default=0,
                        help="[webcam] Camera index (default: 0)")

    # video
    parser.add_argument("--video",      default=None,
                        help="[video] Path to input video file")
    parser.add_argument("--output",     default=None,
                        help="[video] Path to save annotated output video")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="[video] Stop after N frames")
    parser.add_argument("--no-preview", action="store_true",
                        help="[video] Disable live preview window")

    # image
    parser.add_argument("--image", default=None,
                        help="[image] Path to input image file")

    args = parser.parse_args()

    # Validate mode-specific required args early
    if args.mode == "video" and not args.video:
        parser.error("--mode video requires --video <path>")
    if args.mode == "image" and not args.image:
        parser.error("--mode image requires --image <path>")

    recognizer = _load_recognizer(args.model, args.conf, args.gpu)

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
