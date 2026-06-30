"""Core nhận dạng biển số tối ưu cho ứng dụng desktop.

Detector YOLOv8 và OCR đều chạy bằng ONNX Runtime trên CPU. Ứng dụng không
phụ thuộc PyTorch/PaddlePaddle và không có khái niệm ROI.
"""

from __future__ import annotations

import ast
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from rapidocr import RapidOCR


PLATE_CLASS_ID = 2
DEFAULT_CONFIDENCE = 0.30

_LETTER_AS_DIGIT = {
    "A": "4", "B": "8", "D": "0", "G": "6", "I": "1", "L": "1",
    "O": "0", "Q": "0", "S": "5", "T": "7", "U": "0", "Z": "2",
}
_DIGIT_AS_LETTER = {
    "0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G",
    "7": "T", "8": "B",
}


def _resource_roots() -> list[Path]:
    roots: list[Path] = []
    if getattr(sys, "frozen", False):
        roots.append(Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)))
        roots.append(Path(sys.executable).parent)
    roots.append(Path(__file__).resolve().parent)
    return roots


def find_best_model() -> str | None:
    """Tìm model ONNX được đóng gói hoặc model trong workspace."""
    relative_candidates = (
        Path("models/best_vietnam_lpr.onnx"),
        Path("weights/best_vietnam_lpr.onnx"),
        Path("runs/detect/vietnam_lpr/yolov8_local/weights/best.onnx"),
    )
    for root in _resource_roots():
        for relative in relative_candidates:
            candidate = root / relative
            if candidate.is_file():
                return str(candidate)
    return None


def clean_plate_text(text: str) -> str:
    """Chuẩn hóa OCR và sửa các nhầm lẫn chữ/số phổ biến."""
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    if len(cleaned) < 5:
        return cleaned

    chars = list(cleaned)
    for index in range(min(2, len(chars))):
        chars[index] = _LETTER_AS_DIGIT.get(chars[index], chars[index])

    if len(chars) > 2:
        chars[2] = _DIGIT_AS_LETTER.get(chars[2], chars[2])

    # Một hoặc hai ký tự sê-ri, sau đó là dãy số.
    digit_start = 4 if len(chars) >= 9 and chars[3].isalpha() else 3
    if digit_start == 4:
        chars[3] = _DIGIT_AS_LETTER.get(chars[3], chars[3])
    for index in range(digit_start, len(chars)):
        chars[index] = _LETTER_AS_DIGIT.get(chars[index], chars[index])
    return "".join(chars)


def is_valid_vietnam_plate(text: str) -> bool:
    """Kiểm tra các dạng biển số Việt Nam thông dụng."""
    value = re.sub(r"[^A-Z0-9]", "", text.upper())
    if not 7 <= len(value) <= 10:
        return False

    patterns = (
        r"\d{2}[A-Z]\d{4,6}",           # ô tô: 30A12345
        r"\d{2}[A-Z]{2}\d{4,6}",       # xe máy: 29AB12345
        r"\d{2}[A-Z][1-9]\d{4,6}",     # xe máy: 29A112345
        r"\d{2,5}(LD|DA|KT|CD|RM|HC|MK|NG|QT|CV|NN)\d{2,6}",
        r"[ABHKQTPCV][A-Z]\d{4,5}",     # quân đội
    )
    return any(re.fullmatch(pattern, value) for pattern in patterns)


def _letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (size - new_width) // 2
    pad_y = (size - new_height) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
    return canvas, scale, pad_x, pad_y


class OnnxPlateDetector:
    """YOLOv8 inference tối giản bằng ONNX Runtime."""

    def __init__(self, model_path: str, confidence: float = DEFAULT_CONFIDENCE):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 2) // 2))
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input = self.session.get_inputs()[0]
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = int(self.input.shape[-1])
        self.confidence = confidence
        self.class_names = self._read_class_names()

    def _read_class_names(self) -> dict[int, str]:
        raw = self.session.get_modelmeta().custom_metadata_map.get("names", "")
        try:
            names = ast.literal_eval(raw)
            return {int(key): str(value) for key, value in names.items()}
        except (SyntaxError, ValueError, AttributeError):
            return {PLATE_CLASS_ID: "plate"}

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        image, scale, pad_x, pad_y = _letterbox(frame, self.input_size)
        tensor = cv2.dnn.blobFromImage(
            image, scalefactor=1.0 / 255.0, size=(self.input_size, self.input_size),
            swapRB=True, crop=False,
        )
        output = self.session.run([self.output_name], {self.input.name: tensor})[0]
        prediction = np.squeeze(output, axis=0)
        if prediction.shape[0] < prediction.shape[1]:
            prediction = prediction.T

        class_scores = prediction[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(prediction)), class_ids]
        keep = (class_ids == PLATE_CLASS_ID) & (scores >= self.confidence)
        prediction = prediction[keep]
        scores = scores[keep]
        if len(prediction) == 0:
            return []

        boxes_xywh = prediction[:, :4]
        boxes_for_nms: list[list[int]] = []
        boxes_xyxy: list[tuple[int, int, int, int]] = []
        frame_h, frame_w = frame.shape[:2]
        for cx, cy, width, height in boxes_xywh:
            x1 = int((cx - width / 2 - pad_x) / scale)
            y1 = int((cy - height / 2 - pad_y) / scale)
            x2 = int((cx + width / 2 - pad_x) / scale)
            y2 = int((cy + height / 2 - pad_y) / scale)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_w - 1, x2), min(frame_h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                boxes_for_nms.append([0, 0, 0, 0])
                boxes_xyxy.append((0, 0, 0, 0))
                continue
            boxes_for_nms.append([x1, y1, x2 - x1, y2 - y1])
            boxes_xyxy.append((x1, y1, x2, y2))

        selected = cv2.dnn.NMSBoxes(boxes_for_nms, scores.tolist(), self.confidence, 0.45)
        if len(selected) == 0:
            return []
        indices = np.asarray(selected).reshape(-1)
        return [(*boxes_xyxy[index], float(scores[index])) for index in indices]


def _preprocess_plate(crop: np.ndarray) -> np.ndarray:
    height, width = crop.shape[:2]
    if width < 320:
        scale = 320 / max(width, 1)
        crop = cv2.resize(
            crop, (320, max(32, round(height * scale))), interpolation=cv2.INTER_CUBIC,
        )
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    sharpened = cv2.addWeighted(gray, 1.5, cv2.GaussianBlur(gray, (0, 0), 2), -0.5, 0)
    enhanced = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    border = max(4, enhanced.shape[1] // 40)
    return cv2.copyMakeBorder(
        enhanced, border, border, border, border, cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


class OnnxPlateOCR:
    def __init__(self):
        # Model small mặc định đã nằm trong wheel RapidOCR, chạy offline.
        self.engine = RapidOCR()

    def read(self, crop: np.ndarray) -> tuple[str, float, str, bool]:
        if crop.size == 0:
            return "", 0.0, "", False
        result = self.engine(_preprocess_plate(crop), use_det=True, use_cls=False, use_rec=True)
        texts = tuple(getattr(result, "txts", ()) or ())
        scores = tuple(getattr(result, "scores", ()) or ())
        if not texts:
            return "", 0.0, "", False
        raw_text = " ".join(str(text) for text in texts).upper().strip()
        confidence = float(np.mean(scores)) if scores else 0.0
        cleaned = clean_plate_text(raw_text)
        return cleaned, confidence, raw_text, is_valid_vietnam_plate(cleaned)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return intersection / (area_a + area_b - intersection)


@dataclass
class _Track:
    bbox: tuple[int, int, int, int]
    text: str = ""
    raw_text: str = ""
    ocr_conf: float = 0.0
    valid: bool = False
    last_seen: float = 0.0
    last_ocr: float = 0.0


class LicensePlateRecognizer:
    """API mức cao cho ảnh tĩnh, video và camera."""

    def __init__(self, model_path: str | None = None, confidence_threshold: float = DEFAULT_CONFIDENCE):
        selected_model = model_path or find_best_model()
        if not selected_model:
            raise FileNotFoundError("Không tìm thấy models/best_vietnam_lpr.onnx")
        self.detector = OnnxPlateDetector(selected_model, confidence_threshold)
        self.ocr = OnnxPlateOCR()
        self._tracks: list[_Track] = []
        self._recent: dict[str, float] = {}

    @staticmethod
    def _crop(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        height, width = frame.shape[:2]
        pad_x = max(8, (x2 - x1) // 10)
        pad_y = max(6, (y2 - y1) // 5)
        return frame[
            max(0, y1 - pad_y):min(height, y2 + pad_y),
            max(0, x1 - pad_x):min(width, x2 + pad_x),
        ]

    def recognize(self, frame: np.ndarray) -> list[dict]:
        recognitions: list[dict] = []
        for x1, y1, x2, y2, detection_conf in self.detector.detect(frame):
            bbox = (x1, y1, x2, y2)
            crop = self._crop(frame, bbox)
            text, ocr_conf, raw_text, valid = self.ocr.read(crop)
            recognitions.append({
                "bbox": bbox,
                "detection_conf": detection_conf,
                "raw_text": raw_text,
                "text": text,
                "ocr_conf": ocr_conf,
                "is_valid": valid,
            })
        return recognitions

    def reset_stream(self) -> None:
        self._tracks.clear()
        self._recent.clear()

    def process_stream_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict], list[dict]]:
        """Nhận dạng frame với cache không gian để tránh OCR lặp lại."""
        now = time.monotonic()
        detections = self.detector.detect(frame)
        active_tracks: list[_Track] = []
        used: set[int] = set()
        results: list[dict] = []
        events: list[dict] = []

        for x1, y1, x2, y2, detection_conf in detections:
            bbox = (x1, y1, x2, y2)
            best_index = -1
            best_iou = 0.0
            for index, track in enumerate(self._tracks):
                if index in used:
                    continue
                overlap = _iou(bbox, track.bbox)
                if overlap > best_iou:
                    best_iou, best_index = overlap, index
            if best_index >= 0 and best_iou >= 0.25:
                track = self._tracks[best_index]
                used.add(best_index)
                track.bbox = bbox
            else:
                track = _Track(bbox=bbox)
            track.last_seen = now

            box_width = x2 - x1
            should_ocr = box_width >= 55 and (
                track.last_ocr == 0.0 or (not track.valid and now - track.last_ocr >= 0.9)
            )
            if should_ocr:
                track.last_ocr = now
                text, confidence, raw_text, valid = self.ocr.read(self._crop(frame, bbox))
                if text and confidence >= track.ocr_conf:
                    track.text = text
                    track.raw_text = raw_text
                    track.ocr_conf = confidence
                    track.valid = valid

            item = {
                "bbox": bbox,
                "detection_conf": detection_conf,
                "raw_text": track.raw_text,
                "text": track.text,
                "ocr_conf": track.ocr_conf,
                "is_valid": track.valid,
            }
            results.append(item)
            active_tracks.append(track)

            accepted = track.valid or (len(track.text) >= 7 and track.ocr_conf >= 0.65)
            last_event = self._recent.get(track.text, 0.0)
            if accepted and track.text and now - last_event >= 8.0:
                events.append(item.copy())
                self._recent[track.text] = now

        self._tracks = active_tracks
        self._recent = {text: seen for text, seen in self._recent.items() if now - seen < 60.0}
        return self.draw_results(frame, results), results, events

    @staticmethod
    def draw_results(frame: np.ndarray, recognitions: list[dict]) -> np.ndarray:
        output = frame.copy()
        for result in recognitions:
            x1, y1, x2, y2 = result["bbox"]
            valid = result.get("is_valid", False)
            color = (36, 214, 107) if valid else (45, 183, 255)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            text = result.get("text") or "Dang doc..."
            confidence = result.get("ocr_conf", 0.0)
            label = f"{text}  {confidence:.0%}" if confidence else text
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            top = max(label_h + 8, y1)
            cv2.rectangle(output, (x1, top - label_h - 8), (x1 + label_w + 8, top), color, -1)
            cv2.putText(output, label, (x1 + 4, top - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, (16, 24, 32), 2, cv2.LINE_AA)
        return output
