"""
Vietnam License Plate Recognition - Test with Static Image
==========================================================
Usage (from repo root):
    python scripts/test_image.py --image path/to/photo.jpg
    python scripts/test_image.py --image photo.jpg --model weights/best_vietnam_lpr.pt --conf 0.4
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import argparse
import cv2
import matplotlib.pyplot as plt

from lpr_pipeline import LicensePlateRecognizer, find_best_model


def test_with_image(recognizer, image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return

    # Auto-resize nếu ảnh quá lớn — YOLO detect tốt nhất khi object chiếm 5-30% frame
    h, w = frame.shape[:2]
    max_dim = 1280
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        print(f"Resized to: {int(w * scale)}x{int(h * scale)}px")

    print(f"Processing: {image_path}  ({frame.shape[1]}x{frame.shape[0]}px)")
    raw = recognizer.yolo.predict(frame, imgsz=1280, conf=0.1, verbose=False)
    print(f"[DEBUG] YOLO raw detect (conf=0.1): {len(raw[0].boxes)} boxes")
    for b in raw[0].boxes:
        print(f"  class={int(b.cls[0])}  conf={float(b.conf[0]):.2f}  box={b.xyxy[0].tolist()}")

    recognitions = recognizer.recognize(frame)
    output = recognizer.draw_results(frame, recognitions)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(cv2.cvtColor(frame,  cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image",    fontsize=12, weight="bold")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Recognition Result", fontsize=12, weight="bold")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

    print(f"\n=== RESULT ({len(recognitions)} plate(s) found) ===")
    for i, rec in enumerate(recognitions):
        status = "[OK]" if rec["is_valid"] else "[?] "
        print(f"\nPlate #{i + 1}: {status}")
        print(f"  Text          : {rec['text']}")
        print(f"  Raw OCR       : {rec['raw_text']}")
        print(f"  OCR confidence: {rec['ocr_conf']:.1%}")
        print(f"  Det confidence: {rec['detection_conf']:.1%}")

    if not recognitions:
        print("No plates detected. Try lowering --conf (e.g. 0.3).")


def main():
    parser = argparse.ArgumentParser(
        description="Test Vietnamese LPR with a static image"
    )
    parser.add_argument("--image",  required=True,
                        help="Path to input image")
    parser.add_argument("--model",  default=None,
                        help="Path to .pt model file")
    parser.add_argument("--conf",   type=float, default=0.5,
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
    test_with_image(recognizer, args.image)


if __name__ == "__main__":
    main()