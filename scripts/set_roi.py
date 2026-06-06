"""
Vietnam LPR — Công Cụ Chọn Vùng ROI Tương Tác
===============================================
Mở camera, kéo chuột chọn vùng ROI, bấm Enter/Space để lưu → config/roi.yaml.

Cách dùng:
    python scripts/set_roi.py
    python scripts/set_roi.py --camera 1
    python scripts/set_roi.py --video path/to/video.mp4   (dùng 1 frame từ video)

Phím tắt:
    Enter / Space  — Xác nhận ROI và lưu
    C              — Chọn lại
    Q / Esc        — Thoát không lưu
"""

from pathlib import Path
import sys
import argparse
import cv2

ROOT = Path(__file__).resolve().parents[1]
ROI_CONFIG = ROOT / "config" / "roi.yaml"


def grab_frame(source):
    """Lấy 1 frame từ camera hoặc file video."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[LỖI] Không thể mở nguồn: {source}")
        sys.exit(1)

    # Đọc vài frame để camera ổn định (tránh frame đen lúc khởi động)
    for _ in range(10):
        ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("[LỖI] Không thể đọc frame từ nguồn.")
        sys.exit(1)

    return frame


def save_roi(x1_pct, y1_pct, x2_pct, y2_pct):
    """Ghi ROI ra file config/roi.yaml."""
    ROI_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "# ROI — tự sinh bởi set_roi.py\n"
        "# Chạy lại `python scripts/set_roi.py` để chọn vùng mới.\n\n"
        "enabled: true\n\n"
        f"x1: {x1_pct:.4f}\n"
        f"y1: {y1_pct:.4f}\n"
        f"x2: {x2_pct:.4f}\n"
        f"y2: {y2_pct:.4f}\n"
    )
    ROI_CONFIG.write_text(content, encoding="utf-8")
    print(f"\n[OK] ROI đã lưu tại: {ROI_CONFIG}")
    print(f"     x1={x1_pct:.2%}  y1={y1_pct:.2%}  x2={x2_pct:.2%}  y2={y2_pct:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Chọn vùng ROI tương tác cho LPR")
    parser.add_argument("--camera", type=int, default=0,
                        help="Chỉ số camera (mặc định: 0)")
    parser.add_argument("--video", default=None,
                        help="Dùng 1 frame từ file video thay vì camera")
    args = parser.parse_args()

    source = args.video if args.video else args.camera
    print(f"[THÔNG TIN] Đang lấy frame từ: {source}")
    frame = grab_frame(source)
    h, w = frame.shape[:2]

    print(f"[THÔNG TIN] Kích thước frame: {w}x{h}")
    print(f"[THÔNG TIN] Kéo chuột để chọn vùng ROI, rồi bấm Enter/Space để lưu.")
    print(f"            Bấm C để chọn lại. Bấm Q/Esc để hủy.")

    window_name = "Chon ROI — Keo de chon, Enter/Space de luu"
    roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    rx, ry, rw, rh = roi
    if rw < 10 or rh < 10:
        print("[ĐÃ HỦY] Không có ROI được chọn (quá nhỏ hoặc đã hủy).")
        return

    # Chuyển pixel → tỷ lệ %
    x1_pct = rx / w
    y1_pct = ry / h
    x2_pct = (rx + rw) / w
    y2_pct = (ry + rh) / h

    save_roi(x1_pct, y1_pct, x2_pct, y2_pct)

    # Xem trước kết quả
    preview = frame.copy()
    cv2.rectangle(preview, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), (0, 0, 0), -1)
    # Làm tối vùng ngoài ROI
    mask = frame.copy()
    mask[:] = (0, 0, 0)
    mask[ry:ry+rh, rx:rx+rw] = frame[ry:ry+rh, rx:rx+rw]
    dimmed = cv2.addWeighted(frame, 0.3, mask, 0.7, 0)
    cv2.rectangle(dimmed, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
    cv2.putText(dimmed, "ROI DA LUU — Bam phim bat ky de dong",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Xem Truoc ROI", dimmed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
