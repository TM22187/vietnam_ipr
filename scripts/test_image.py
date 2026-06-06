"""
Nhận Dạng Biển Số Xe Việt Nam - Kiểm Tra Với Ảnh Tĩnh
======================================================
Cách dùng (từ thư mục gốc dự án):
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
        print(f"[LỖI] Không thể đọc ảnh: {image_path}")
        return

    # Tự thu nhỏ nếu ảnh quá lớn — YOLO detect tốt nhất khi object chiếm 5-30% frame
    h, w = frame.shape[:2]
    max_dim = 1280
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        print(f"Đã thu nhỏ ảnh còn: {int(w * scale)}x{int(h * scale)}px")

    print(f"Đang xử lý: {image_path}  ({frame.shape[1]}x{frame.shape[0]}px)")
    raw = recognizer.yolo.predict(frame, imgsz=1280, conf=0.1, verbose=False)
    print(f"[DEBUG] YOLO phát hiện thô (conf=0.1): {len(raw[0].boxes)} box")
    for b in raw[0].boxes:
        print(f"  class={int(b.cls[0])}  conf={float(b.conf[0]):.2f}  box={b.xyxy[0].tolist()}")

    recognitions = recognizer.recognize(frame)
    output = recognizer.draw_results(frame, recognitions)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(cv2.cvtColor(frame,  cv2.COLOR_BGR2RGB))
    axes[0].set_title("Ảnh Gốc",    fontsize=12, weight="bold")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Kết Quả Nhận Dạng", fontsize=12, weight="bold")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

    print(f"\n=== KẾT QUẢ ({len(recognitions)} biển số tìm thấy) ===")
    for i, rec in enumerate(recognitions):
        status = "[OK]" if rec["is_valid"] else "[?] "
        print(f"\nBiển số #{i + 1}: {status}")
        print(f"  Văn bản          : {rec['text']}")
        print(f"  OCR thô          : {rec['raw_text']}")
        print(f"  Độ tự tin OCR    : {rec['ocr_conf']:.1%}")
        print(f"  Độ tự tin detect : {rec['detection_conf']:.1%}")

    if not recognitions:
        print("Không phát hiện biển số nào. Thử giảm --conf (ví dụ: 0.3).")


def main():
    parser = argparse.ArgumentParser(
        description="Kiểm tra nhận dạng biển số xe Việt Nam với ảnh tĩnh"
    )
    parser.add_argument("--image",  required=True,
                        help="Đường dẫn ảnh đầu vào")
    parser.add_argument("--model",  default=None,
                        help="Đường dẫn file model .pt")
    parser.add_argument("--conf",   type=float, default=0.5,
                        help="Ngưỡng confidence detection")
    parser.add_argument("--gpu",    action="store_true",
                        help="Dùng GPU cho YOLO (PaddleOCR luôn CPU)")
    args = parser.parse_args()

    model_path = args.model or find_best_model()
    if not model_path or not os.path.exists(model_path):
        print(
            "[LỖI] Không tìm thấy file model. Đặt weights vào thư mục weights/ "
            "hoặc truyền --model path/to/best.pt")
        return

    recognizer = LicensePlateRecognizer(
        model_path, confidence_threshold=args.conf, use_gpu=args.gpu)
    test_with_image(recognizer, args.image)


if __name__ == "__main__":
    main()