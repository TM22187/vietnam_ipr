"""
Nhận Dạng Biển Số Xe Việt Nam - Pipeline Cốt Lõi

Cung cấp các hàm hỗ trợ OCR và class LicensePlateRecognizer
kết hợp YOLOv8 (phát hiện) + PaddleOCR (đọc ký tự).

Cách dùng (dạng module):
    from lpr_pipeline import LicensePlateRecognizer
    recognizer = LicensePlateRecognizer("weights/best_vietnam_lpr.pt")
    results = recognizer.recognize(frame)
"""

import os
import re
import glob
import time
import logging
import cv2
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO

logger = logging.getLogger("lpr")

# Thư mục gốc của dự án (thư mục chứa file này)
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# ID class biển số trong dataset (0=ô tô, 1=xe máy, 2=biển số)
PLATE_CLASS_ID = 2

# Ngưỡng confidence OCR tối thiểu để chấp nhận kết quả
MIN_OCR_CONFIDENCE = 0.0

# Độ lệch chuẩn Laplacian tối thiểu để coi ảnh crop "đủ nét" cho OCR
MIN_BLUR_SCORE = 50.0

# Bảng hoán đổi ký tự OCR hay nhầm (áp dụng bên trong fix_ocr_mistakes).
_OCR_LETTER_AS_DIGIT = {
    "A": "4", "B": "8", "D": "0", "G": "6",
    "I": "1", "L": "1", "O": "0", "Q": "0",
    "S": "5", "T": "7", "U": "0", "Z": "2",
}
_OCR_DIGIT_AS_LETTER = {
    "0": "O", "1": "I", "2": "Z", "4": "A",
    "5": "S", "6": "G", "7": "T", "8": "B",
}


def _sharpen(img):
    """Làm sắc nét ảnh bằng Unsharp-mask để cạnh ký tự rõ hơn."""
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)


def preprocess_plate(plate_img):
    h, w = plate_img.shape[:2]

    # 1. Bỏ qua deskew vì camera cổng thường cố định góc chụp

    # --- Thu nhỏ ảnh để tiết kiệm CPU ---
    # Ảnh gốc khoảng 120-160px là đủ để OCR đọc nét.
    # Phóng to lên 320px khiến CPU tính gấp 4 lần không cần thiết!
    target_w = 160  # Thay vì max(w, 320)

    # Dùng INTER_AREA khi thu nhỏ cho chất lượng tốt hơn INTER_CUBIC
    plate_img = cv2.resize(plate_img, (target_w, int(h * target_w / w)),
                           interpolation=cv2.INTER_AREA)

    # 2. Thêm viền đen xung quanh (tránh ký tự sát mép bị cắt)
    pad_b = int(target_w * 0.05)
    plate_img = cv2.copyMakeBorder(plate_img, pad_b, pad_b, pad_b, pad_b,
                                   cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # 3. Cân bằng histogram cục bộ (CLAHE) + làm sắc nét
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    sharpened = _sharpen(plate_img)
    gray_sharp = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    v1 = cv2.cvtColor(clahe.apply(gray_sharp), cv2.COLOR_GRAY2BGR)

    return [v1]


def read_plate_text(ocr_engine, plate_img):
    """
    Đọc text từ ảnh crop biển số bằng PaddleOCR.
    Thử nhiều phiên bản tiền xử lý, trả về kết quả có confidence cao nhất.

    Tham số:
        ocr_engine: Instance PaddleOCR
        plate_img:  numpy array (BGR)

    Trả về:
        tuple (text: str, confidence: float)
    """
    best_text = ""
    best_conf = 0.0

    for img_bgr in preprocess_plate(plate_img):
        try:
            result = ocr_engine.predict(img_bgr)
        except Exception as e:
            print(f"[Cảnh báo OCR] {e}")
            continue

        if not result or not result[0]:
            continue

        res = result[0]
        texts = res.get("rec_texts", [])
        confs = res.get("rec_scores", [])

        if not texts:
            continue

        text = " ".join(texts).upper().strip()
        conf = float(np.mean(confs)) if confs else 0.0

        if conf > best_conf:
            best_text = text
            best_conf = conf

    return best_text, best_conf


def fix_ocr_mistakes(text):
    """
    Sửa lỗi OCR theo quy tắc dựa trên cấu trúc biển số Việt Nam
    (mã tỉnh là chữ số, ký tự sê-ri là chữ cái, phần số cuối là chữ số).
    """
    if len(text) < 7:
        return text

    chars = list(text)

    # Vị trí 0-1: phải là SỐ (mã tỉnh)
    for i in range(min(2, len(chars))):
        if chars[i] in _OCR_LETTER_AS_DIGIT:
            chars[i] = _OCR_LETTER_AS_DIGIT[chars[i]]

    # Vị trí 2: phải là CHỮ CÁI (ký tự sê-ri)
    if len(chars) > 2 and chars[2] in _OCR_DIGIT_AS_LETTER:
        chars[2] = _OCR_DIGIT_AS_LETTER[chars[2]]

    if len(chars) > 3:
        if len(chars) >= 9:
            # Biển 9 ký tự: vị trí 3 có thể là chữ O trong sê-ri đặc biệt
            if chars[3] == "0":
                chars[3] = "O"
            start_digit = 4
        else:
            if chars[3] in _OCR_LETTER_AS_DIGIT:
                chars[3] = _OCR_LETTER_AS_DIGIT[chars[3]]
            start_digit = 3

        # Phần số cuối: phải là CHỮ SỐ
        for i in range(start_digit, len(chars)):
            if chars[i] in _OCR_LETTER_AS_DIGIT:
                chars[i] = _OCR_LETTER_AS_DIGIT[chars[i]]

    return "".join(chars)


def clean_plate_text(text):
    """Loại bỏ ký tự không phải chữ/số và áp dụng sửa lỗi OCR."""
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    return fix_ocr_mistakes(cleaned)


# ─────────────────────────────────────────────
# LOGIC KIỂM TRA HỢP LỆ (TT 79/2024, TT 51/2025 & Thực tế)
# ─────────────────────────────────────────────

# Bảng chữ cái hợp lệ theo Luật (Bỏ I, O, Q, R, W)
_L = r"[ABCDEFGHKLMNPSTUVXYZ]"

# Mã đặc biệt (LD, DA, MK...) và mã ngoại giao (NG, QT, CV, NN)
_SPEC = r"(LD|DA|KT|CD|RM|HC|MK|TĐ|MĐ|CT|LB|R|NG|QT|CV|NN)"

# Mã Quân đội bắt đầu bằng các chữ quy định
_ARMY = r"([ABHKQTPCV][A-Z])"


def is_valid_vietnam_plate(text):
    """
    Kiểm tra theo Thông tư 79/2024, Thông tư 51/2025 và thực tế lưu thông.
    Bao gồm cơ chế "Nới lỏng" (Partial read) để không bỏ sót dữ liệu hiển thị.
    """
    text = re.sub(r"[^A-Z0-9]", "", text.upper())

    # --- 1. LỌC RÁC: Dưới 4 ký tự chắc chắn là nhiễu → Loại bỏ ---
    if len(text) < 4:
        return False

    # --- 2. KIỂM TRA CHUẨN FORMAT (Luật Mới + Luật Cũ) ---
    is_perfect_match = False

    # Trường hợp A: Xe Quân Đội (2 chữ cái quân đội + 4-5 số)
    # Ví dụ: KP1234, TM12345
    if re.match(rf"^{_ARMY}\d{{4,5}}$", text):
        is_perfect_match = True

    # Trường hợp B: Xe Dân Sự / Nhà Nước / Ngoại Giao (bắt đầu bằng 2 số tỉnh 11-99)
    elif len(text) >= 6 and text[:2].isdigit() and 11 <= int(text[:2]) <= 99:

        # B.1: Ô tô (1 chữ cái hợp lệ + 4 hoặc 5 số) — Ví dụ: 30A12345, 29A1234
        if re.match(rf"^\d{{2}}{_L}\d{{4,6}}$", text):
            is_perfect_match = True

        # B.2: Xe máy (2 chữ cái hợp lệ + 4 hoặc 5 số) — Ví dụ: 29AB12345
        elif re.match(rf"^\d{{2}}{_L}{{2}}\d{{4,6}}$", text):
            is_perfect_match = True

        # B.3: Xe máy dùng 1 chữ 1 số — Ví dụ: 29A112345
        elif re.match(rf"^\d{{2}}{_L}[1-9]\d{{4,6}}$", text):
            is_perfect_match = True

        # B.4: Xe mang ký hiệu đặc biệt hoặc ngoại giao — Ví dụ: 29LD12345, 29123NG45
        elif re.match(rf"^\d{{2,5}}{_SPEC}\d{{2,6}}$", text):
            is_perfect_match = True

    # Đúng định dạng 100% → hợp lệ
    if is_perfect_match:
        return True

    # Nếu không chuẩn 100% (AI đọc sót chữ) nhưng chuỗi đủ dài >= 5 ký tự
    # → Vẫn cho qua để hiển thị lên màn hình (để mắt người tự luận).
    if len(text) >= 5:
        return True

    return False

# ─────────────────────────────────────────────
# Hàm tìm file model tốt nhất
# ─────────────────────────────────────────────


def find_best_model():
    """Tìm file weights best.pt theo thứ tự ưu tiên: weights/, root, runs/detect/**/."""
    wd = os.path.join(REPO_ROOT, "weights")
    from_package = glob.glob(
        os.path.join(REPO_ROOT, "runs", "detect", "**", "best.pt"),
        recursive=True,
    )
    candidates = [
        os.path.join(wd, "best_vietnam_lpr.pt"),
        os.path.join(wd, "best.pt"),
        os.path.join(REPO_ROOT, "best_vietnam_lpr.pt"),
        os.path.join(REPO_ROOT, "best.pt"),
        *from_package,
        "best_vietnam_lpr.pt",
        "best.pt",
        *glob.glob("runs/detect/**/best.pt", recursive=True),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.normpath(path)
    return None


# ─────────────────────────────────────────────
# LicensePlateRecognizer
# ─────────────────────────────────────────────

class LicensePlateRecognizer:
    """
    Nhận dạng biển số xe Việt Nam end-to-end.
    Kết hợp YOLOv8 (phát hiện) và PaddleOCR (đọc ký tự).

    GHI CHÚ: YOLO chạy GPU (truyền device khi gọi track/predict).
              PaddleOCR luôn chạy CPU để tránh conflict CUDA context với PyTorch.
              enable_mkldnn=True tăng tốc PaddleOCR trên CPU ~30-40%.
    """

    def __init__(self, yolo_model_path, confidence_threshold=0.3,
                 plate_class_id=PLATE_CLASS_ID, use_gpu=False,
                 roi=None):
        """
        Tham số:
            yolo_model_path:      đường dẫn file weights .pt
            confidence_threshold: ngưỡng confidence detection tối thiểu (0–1)
            plate_class_id:       index class 'biển số' trong dataset
            use_gpu:              True để YOLO dùng GPU (PaddleOCR luôn CPU)
            roi:                  (x1%, y1%, x2%, y2%) tỷ lệ 0.0–1.0, hoặc None
        """
        logger.info("Đang tải model YOLOv8...")
        self.yolo = YOLO(yolo_model_path)
        self.use_gpu = use_gpu

        logger.info("Đang tải PaddleOCR...")
        # PaddleOCR luôn chạy CPU — tránh conflict CUDA context với YOLO/PyTorch.
        # enable_mkldnn=True bù lại bằng cách tăng tốc CPU Intel ~30-40%.
        self.ocr = PaddleOCR(
            lang="en",
            device="cpu",
            enable_mkldnn=False,                 # Giữ False để tránh crash ngầm
            cpu_threads=4,                       # 4 luồng, nhường CPU cho camera thread
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        self.conf_threshold = confidence_threshold
        self.plate_class_id = plate_class_id

        # ROI — tỷ lệ % (0.0–1.0) để không phụ thuộc vào độ phân giải camera
        self.roi = roi
        if roi:
            logger.info(f"ROI đang bật: x=[{roi[0]:.0%}–{roi[2]:.0%}], "
                        f"y=[{roi[1]:.0%}–{roi[3]:.0%}]")
        else:
            logger.info("ROI: tắt (phát hiện toàn khung hình)")

        logger.info("[HOÀN TẤT] Pipeline sẵn sàng!")

    # ── Các hàm hỗ trợ ROI ──────────────────────────────────────

    def set_roi(self, roi):
        """Đặt ROI dạng (x1_pct, y1_pct, x2_pct, y2_pct) trong khoảng 0.0–1.0."""
        self.roi = roi
        if roi:
            logger.info(f"ROI đã cập nhật: {roi}")
        else:
            logger.info("ROI đã xóa")

    def _is_inside_roi(self, x1, y1, x2, y2, frame_w, frame_h):
        """Kiểm tra xem tâm bounding box có nằm trong vùng ROI không."""
        if self.roi is None:
            return True
        rx1 = int(self.roi[0] * frame_w)
        ry1 = int(self.roi[1] * frame_h)
        rx2 = int(self.roi[2] * frame_w)
        ry2 = int(self.roi[3] * frame_h)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

    def get_roi_pixels(self, frame_w, frame_h):
        """Trả về ROI dạng tọa độ pixel (x1, y1, x2, y2) hoặc None."""
        if self.roi is None:
            return None
        return (
            int(self.roi[0] * frame_w), int(self.roi[1] * frame_h),
            int(self.roi[2] * frame_w), int(self.roi[3] * frame_h),
        )

    def detect_plates(self, frame):
        """
        Chạy YOLOv8 trên frame và trả về danh sách bounding box biển số.
        Các biển số có tâm nằm ngoài ROI sẽ bị bỏ qua.

        Trả về:
            list of (x1, y1, x2, y2, confidence)
        """
        fh, fw = frame.shape[:2]
        results = self.yolo(frame, conf=self.conf_threshold, verbose=False)
        plates = []
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) != self.plate_class_id:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])

                # ── Lọc ROI ──
                if not self._is_inside_roi(x1, y1, x2, y2, fw, fh):
                    continue

                plates.append([x1, y1, x2, y2, conf])

        # Gộp các box chồng lấp
        merged = []
        while plates:
            base = plates.pop(0)
            base_x1, base_y1, base_x2, base_y2, base_conf = base

            i = 0
            while i < len(plates):
                x1, y1, x2, y2, conf = plates[i]
                ix1, iy1 = max(base_x1, x1), max(base_y1, y1)
                ix2, iy2 = min(base_x2, x2), min(base_y2, y2)
                inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area1 = (base_x2 - base_x1) * (base_y2 - base_y1)
                area2 = (x2 - x1) * (y2 - y1)
                if inter_area > 0.3 * min(area1, area2):
                    base_x1, base_y1 = min(base_x1, x1), min(base_y1, y1)
                    base_x2, base_y2 = max(base_x2, x2), max(base_y2, y2)
                    base_conf = max(base_conf, conf)
                    plates.pop(i)
                else:
                    i += 1
            merged.append((base_x1, base_y1, base_x2, base_y2, base_conf))

        return merged

    def ocr_plate_crop(self, crop):
        """
        Chạy OCR trên ảnh crop biển số. Trả về (text_đã_làm_sạch, conf, raw, hợp_lệ).
        Đây là bước CHẬM (~100-300ms mỗi biển số).
        """
        if crop.size == 0:
            return "", 0.0, "", False

        raw_text, ocr_conf = read_plate_text(self.ocr, crop)
        if ocr_conf < MIN_OCR_CONFIDENCE:
            return "", 0.0, raw_text, False

        cleaned = clean_plate_text(raw_text)
        valid = is_valid_vietnam_plate(cleaned)
        return cleaned, ocr_conf, raw_text, valid

    def recognize(self, frame):
        """
        Phát hiện biển số và đọc ký tự (single-shot, dùng cho ảnh tĩnh).

        Trả về:
            list các dict với các key:
                bbox, detection_conf, raw_text, text, ocr_conf,
                is_valid, plate_crop
        """
        recognitions = []
        h, w = frame.shape[:2]

        for (x1, y1, x2, y2, det_conf) in self.detect_plates(frame):
            pad = 20
            crop = frame[max(0, y1 - pad): min(h, y2 + pad),
                         max(0, x1 - pad): min(w, x2 + pad)]
            if crop.size == 0:
                continue

            cleaned, ocr_conf, raw_text, valid = self.ocr_plate_crop(crop)
            if not cleaned:
                continue

            recognitions.append({
                "bbox":           (x1, y1, x2, y2),
                "detection_conf": det_conf,
                "raw_text":       raw_text,
                "text":           cleaned,
                "ocr_conf":       ocr_conf,
                "is_valid":       valid,
                "plate_crop":     crop,
            })

        return recognitions

    def draw_results(self, frame, recognitions):
        """
        Vẽ bounding box và text biển số lên bản sao của frame.

        Trả về:
            ảnh đã chú thích (numpy array, BGR)
        """
        output = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        for rec in recognitions:
            x1, y1, x2, y2 = rec["bbox"]
            text = rec["text"]
            ocr_conf = rec["ocr_conf"]
            # Xanh lá = hợp lệ, vàng = chưa xác định
            color = (0, 255, 0) if rec.get("is_valid", True) else (0, 215, 255)

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            if not text:
                label = "..."
                label_color = (180, 180, 180)
            else:
                label = f"{text} ({ocr_conf:.0%})"
                label_color = (0, 0, 0)

            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            label_y = max(y1 - 10, th + 5)
            cv2.rectangle(output,
                          (x1, label_y - th - 5),
                          (x1 + tw, label_y + 3),
                          color, -1)
            cv2.putText(output, label, (x1, label_y),
                        font, font_scale, label_color, thickness, cv2.LINE_AA)

        return output
