"""
Vietnam License Plate Recognition - Core Pipeline

Provides OCR helper functions and the LicensePlateRecognizer class
that combines YOLOv8 (detection) + PaddleOCR (character reading).

Usage (as a module):
    from lpr_pipeline import LicensePlateRecognizer
    recognizer = LicensePlateRecognizer("weights/best_vietnam_lpr.pt")
    results = recognizer.recognize(frame)
"""

import os
import re
import glob
import queue
import time
import threading
from collections import Counter, deque
from difflib import SequenceMatcher
import cv2
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO

# Repository root (folder containing this file)
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Plate class ID in the dataset (0=car, 1=motorcycle, 2=plate)
PLATE_CLASS_ID = 2

# Minimum OCR confidence to accept a reading
MIN_OCR_CONFIDENCE = 0.0

# Minimum Laplacian variance to consider a plate crop "sharp enough" for OCR
MIN_BLUR_SCORE = 50.0

# Glyph swaps generic OCR often makes on plates (applied only inside fix_ocr_mistakes).
_OCR_LETTER_AS_DIGIT = {
    "A": "4", "B": "8", "D": "0", "G": "6",
    "I": "1", "L": "1", "O": "0", "Q": "0",
    "S": "5", "T": "7", "U": "0", "Z": "2",
}
_OCR_DIGIT_AS_LETTER = {
    "0": "O", "1": "I", "2": "Z", "4": "A",
    "5": "S", "6": "G", "7": "T", "8": "B",
}


def estimate_blur(img):
    """Return Laplacian variance — higher = sharper. Below ~50 is blurry."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(
        img.shape) == 3 else img
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# OCR Helper Functions


def deskew_plate(img):
    """Correct slight rotation using minAreaRect on foreground pixels."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))[:, ::-1]  # (row,col)→(x,y)
    if len(coords) < 20:
        return img
    angle = cv2.minAreaRect(coords)[2]
    if angle < -45:
        angle += 90
    if abs(angle) < 1.0:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _sharpen(img):
    """Unsharp-mask sharpening to make character edges crisper."""
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)


def preprocess_plate(plate_img):
    """
    Preprocess license plate image before OCR.
    Returns one variant: Sharpen + CLAHE (fast, works well for most plates).

    Args:
        plate_img: numpy array (BGR, from OpenCV)

    Returns:
        list of one processed image (BGR)
    """
    h, w = plate_img.shape[:2]

    # 1. Fix tilt
    plate_img = deskew_plate(plate_img)

    # 2. Upscale to at least 320px wide
    target_w = max(w, 320)
    plate_img = cv2.resize(plate_img, (target_w, int(h * target_w / w)),
                           interpolation=cv2.INTER_CUBIC)

    # 3. Add explicit padding (black border)
    # Crucial for close-up plates: if the plate touches the frame edges, the YOLO crop 
    # lacks a margin. PaddleOCR's DBNet often fails to detect text touching the image edge.
    pad_b = int(target_w * 0.05)
    plate_img = cv2.copyMakeBorder(plate_img, pad_b, pad_b, pad_b, pad_b, 
                                   cv2.BORDER_CONSTANT, value=[0, 0, 0])

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    sharpened = _sharpen(plate_img)
    gray_sharp = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    v1 = cv2.cvtColor(clahe.apply(gray_sharp), cv2.COLOR_GRAY2BGR)

    return [v1]


def read_plate_text(ocr_engine, plate_img):
    """
    Read text from a license plate crop using PaddleOCR.
    Tries multiple preprocessing versions, returns the highest-confidence result.

    Args:
        ocr_engine: PaddleOCR instance
        plate_img:  numpy array (BGR)

    Returns:
        tuple (text: str, confidence: float)
    """
    best_text = ""
    best_conf = 0.0

    for img_bgr in preprocess_plate(plate_img):
        try:
            result = ocr_engine.predict(img_bgr)
        except Exception as e:
            print(f"[OCR warning] {e}")
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
    Rule-based cleanup using Vietnamese plate layout (province digits,
    series letters, registration digits).

    When this helps:
        Quick win after a generic Latin OCR engine — fixes frequent O/0,
        I/1, etc., when the string length roughly matches car vs bike plates.

    When this is NOT “best practice” alone:
        It can overwrite rare real sequences and ignores ambiguity (e.g.
        legitimate letters in unusual plates). Stronger pipelines combine:
        sharper crops, recognition tuned on plates or Vietnamese text,
        constrained decoding / beam search against regex, or voting across
        frames (already done elsewhere in this project).

    This layer is intentional “cheap polish”, not a full OCR replacement.
    """
    if len(text) < 7:
        return text

    chars = list(text)

    for i in range(min(2, len(chars))):
        if chars[i] in _OCR_LETTER_AS_DIGIT:
            chars[i] = _OCR_LETTER_AS_DIGIT[chars[i]]

    if len(chars) > 2 and chars[2] in _OCR_DIGIT_AS_LETTER:
        chars[2] = _OCR_DIGIT_AS_LETTER[chars[2]]

    if len(chars) > 3:
        if len(chars) >= 9:
            # Position 3 is the 2nd series char: a letter (new motorcycle 2-letter
            # series) or digit 1-9 (govt car / old motorcycle letter+digit series).
            # '0' is never a valid series digit, so treat it as misread 'O'.
            if chars[3] == "0":
                chars[3] = "O"
            start_digit = 4
        else:
            # 8-char plate: position 3 is the first registration digit
            if chars[3] in _OCR_LETTER_AS_DIGIT:
                chars[3] = _OCR_LETTER_AS_DIGIT[chars[3]]
            start_digit = 3

        for i in range(start_digit, len(chars)):
            if chars[i] in _OCR_LETTER_AS_DIGIT:
                chars[i] = _OCR_LETTER_AS_DIGIT[chars[i]]

    return "".join(chars)


def clean_plate_text(text):
    """Strip non-alphanumeric chars and apply OCR mistake correction."""
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    return fix_ocr_mistakes(cleaned)


# Valide series TT 79/2024 — loại I, O, Q, R, W (not used for seri)
_VALID_SERIES_CHARS = set("ABCDEFGHKMNPSTUVXYZ")

# Forbidden moto series (circular 79/2024, excludes confusing combos and historically rare ones)
_FORBIDDEN_MOTO_SERIES = {"CD", "CT", "DA", "HC", "LB", "LD", "MK"}

# Valid province codes: 11–99, excluding 13 (not in use) and reserved numbers
_RESERVED_PROVINCE = {10, 13, 42, 44, 45, 46, 87, 91, 96}
_VALID_PROVINCE = {f"{i:02d}" for i in range(11, 100) if i not in _RESERVED_PROVINCE}

_S = "[ABCDEFGHKMNPSTUVXYZ]"  # 1-letter series (car) or first letter of 2-letter series (moto)

def is_valid_vietnam_plate(text):
    """
    Return True only if text matches a Vietnamese plate format (Circular 79/2024).

    Formats (registration = 5 digits 000.01-999.99, dots stripped by caller):
      DD[L]DDDDD       (8 chars) - private car (white/yellow plate)  e.g. 29A12345
      DD[L][1-9]DDDDD  (9 chars) - govt car (blue plate) or
                                    old motorcycle (letter+digit series) e.g. 29A112345
      DD[LL]DDDDD      (9 chars) - new motorcycle (2-letter series)   e.g. 29AB12345
    4-digit registrations accepted too (pre-2025 plates still in circulation).

    Stricter than before:
      - Province code must be a real issued code (11-99 minus reserved/unissued)
      - Series letters restricted to the 20 chars actually used (no I,O,Q,R,W)
      - Forbidden 2-letter moto series (CD, CT, DA, HC, LB, LD, MK) rejected
    """
    text = re.sub(r"[^A-Z0-9]", "", text.upper())

    # Check province code first (2 digits, must be in valid set)
    if len(text) < 8 or text[:2] not in _VALID_PROVINCE:
        return False

    # 8 digits: private car (1-letter series) — DD[L]DDDDD
    if re.match(rf"^\d{{2}}{_S}\d{{4,5}}$", text):
        return True

    # 9 digits: govt car (1-letter + digit series) — DD[L][1-9]DDDDD
    if re.match(rf"^\d{{2}}{_S}[1-9]\d{{4,5}}$", text):
        return True

    # 9 digits: new motorcycle (2-letter series) — DD[LL]DDDDD, with forbidden series excluded
    if re.match(rf"^\d{{2}}{_S}{{2}}\d{{4,5}}$", text):
        series = text[2:4]
        if series in _FORBIDDEN_MOTO_SERIES:
            return False
        return True

    return False


# ─────────────────────────────────────────────
# Temporal Smoother
# ─────────────────────────────────────────────

class PlateTracker:
    """
    Stabilizes OCR output by voting across the last N frames.

    Used after TrackedPlateCache: smooths noisy per-read OCR and keeps text
    stable when the same plate moves (bbox matched by IoU, not only track id).

    For each detected plate region, keeps a rolling window of OCR readings.
    The displayed text is the most frequently seen valid result in that window.
    This eliminates the "jumping" effect caused by per-frame OCR noise.
    """

    def __init__(self, window=30, min_votes=4, stale_limit=45):
        """
        Args:
            window:      number of recent OCR results to remember per plate
            min_votes:   minimum similar results required before text is "locked"
            stale_limit: frames a plate can be missing before history is deleted
        """
        self.window = window
        self.min_votes = min_votes
        self.stale_limit = stale_limit
        # slot_id -> deque of (text, conf, bbox) tuples
        self._history: dict[int, deque] = {}
        # slot_id -> (text, conf) — last confirmed stable result per plate
        self._last_stable: dict[int, tuple] = {}
        # slot_id -> int — consecutive frames the plate has been missing
        self._cooldown: dict[int, int] = {}

    @staticmethod
    def _iou(a, b):
        """Intersection-over-Union of two boxes (x1,y1,x2,y2)."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / ua if ua > 0 else 0.0

    @staticmethod
    def _center_dist_ratio(a, b):
        """Ratio of center distance to avg box diagonal. Lower = closer."""
        cx_a, cy_a = (a[0]+a[2])/2, (a[1]+a[3])/2
        cx_b, cy_b = (b[0]+b[2])/2, (b[1]+b[3])/2
        dist = ((cx_a - cx_b)**2 + (cy_a - cy_b)**2) ** 0.5
        diag_a = ((a[2]-a[0])**2 + (a[3]-a[1])**2) ** 0.5
        diag_b = ((b[2]-b[0])**2 + (b[3]-b[1])**2) ** 0.5
        avg_diag = (diag_a + diag_b) / 2
        return dist / avg_diag if avg_diag > 0 else 999.0

    def _match_slot(self, bbox, current_slots):
        """Find the history slot whose last bbox best matches this one.
        Uses IOU first; falls back to center-distance for fast-moving plates."""
        best_id, best_iou = None, 0.15
        for slot_id, (last_bbox, _) in current_slots.items():
            iou = self._iou(bbox, last_bbox)
            if iou > best_iou:
                best_iou = iou
                best_id = slot_id

        # Fallback: if no IOU match, try center-distance matching
        if best_id is None:
            best_dist_id, best_ratio = None, 1.2
            for slot_id, (last_bbox, _) in current_slots.items():
                ratio = self._center_dist_ratio(bbox, last_bbox)
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_dist_id = slot_id
            best_id = best_dist_id

        return best_id

    @staticmethod
    def _similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()

    def _fuzzy_vote(self, entries):
        """
        Group texts with >= 75% similarity, return the largest group's
        (best_text, vote_count, avg_confidence).
        Handles OCR alternating between '29A1234' / '29A12345'.
        """
        candidates = [(t, c) for t, c, _ in entries if t and len(t) >= 5]
        if not candidates:
            return "", 0, 0.0

        groups: list = []
        for t, c in candidates:
            placed = False
            for g in groups:
                if self._similarity(t, g[0][0]) >= 0.75:
                    g.append((t, c))
                    placed = True
                    break
            if not placed:
                groups.append([(t, c)])

        best_group = max(groups, key=len)
        votes = len(best_group)
        counter = Counter(t for t, _ in best_group)
        best_text = counter.most_common(1)[0][0]
        avg_conf = float(np.mean([c for t, c in best_group
                                  if t == best_text]))
        return best_text, votes, avg_conf

    def update(self, recognitions):
        """
        Feed this frame's raw recognitions; returns smoothed recognitions.

        Returns:
            list of dicts with 'text' replaced by voted stable text
            and extra key 'stable' (bool).
        """
        current_slots = {}
        for sid, dq in self._history.items():
            if dq:
                last = dq[-1]
                current_slots[sid] = (last[2], last)

        used_slots = set()
        smoothed = []
        next_id = max(self._history.keys(), default=-1) + 1

        for rec in recognitions:
            bbox = rec["bbox"]
            text = rec["text"]
            conf = rec["ocr_conf"]

            slot_id = self._match_slot(bbox, current_slots)
            if slot_id is None or slot_id in used_slots:
                slot_id = next_id
                next_id += 1
                self._history[slot_id] = deque(maxlen=self.window)

            used_slots.add(slot_id)
            self._cooldown[slot_id] = 0   # plate is visible → reset cooldown
            self._history[slot_id].append((text, conf, bbox))

            entries = list(self._history[slot_id])
            best_text, votes, avg_conf = self._fuzzy_vote(entries)

            stable = votes >= self.min_votes
            if stable:
                # New confirmed reading — lock it in
                self._last_stable[slot_id] = (best_text, avg_conf)
                display, display_conf = best_text, avg_conf
            elif slot_id in self._last_stable:
                # Not enough votes yet — keep showing last confirmed text
                display, display_conf = self._last_stable[slot_id]
                # treat as stable visually (frozen on last good result)
                stable = True
            else:
                # No stable history at all yet — show nothing
                display, display_conf = "", 0.0

            out = dict(rec)
            out["text"] = display
            out["ocr_conf"] = display_conf
            out["stable"] = stable
            smoothed.append(out)

        # ── Stale slot cleanup WITH cooldown ─────────────────────────────
        # Instead of deleting immediately, give each missing plate
        # `stale_limit` frames before wiping its history.
        stale_to_delete = []
        for sid in list(self._history.keys()):
            if sid not in used_slots:
                self._cooldown[sid] = self._cooldown.get(sid, 0) + 1
                if self._cooldown[sid] > self.stale_limit:
                    stale_to_delete.append(sid)

        for sid in stale_to_delete:
            del self._history[sid]
            self._last_stable.pop(sid, None)
            self._cooldown.pop(sid, None)

        return smoothed


def _resolve_tracker_yaml():
    """Prefer config/bytetrack_lpr.yaml, else repo-root legacy path, else Ultralytics default."""
    for rel in ("config/bytetrack_lpr.yaml", "bytetrack_lpr.yaml"):
        p = os.path.join(REPO_ROOT, *rel.split("/"))
        if os.path.isfile(p):
            return p
    return "bytetrack.yaml"


def match_vehicle_for_plate(plate_bbox, vehicles):
    """
    Link a plate box to the most plausible vehicle (plate center inside box,
    else highest IoU). Returns vehicle track_id or None.
    """
    if not vehicles:
        return None
    px = (plate_bbox[0] + plate_bbox[2]) / 2
    py = (plate_bbox[1] + plate_bbox[3]) / 2
    best_tid = None
    best_score = -1.0
    for v in vehicles:
        x1, y1, x2, y2 = v["bbox"]
        inside = x1 <= px <= x2 and y1 <= py <= y2
        iou = PlateTracker._iou(plate_bbox, v["bbox"])
        score = max(iou, 0.2 if inside else 0.0)
        if score > best_score:
            best_score = score
            best_tid = v["track_id"]
    return best_tid


def prepare_plate_recognitions(tracking, plate_cache):
    """
    Merge live plate boxes with OCR cache into rows for PlateTracker.update().
    Each row includes vehicle_track_id for consistent coloring with draw_tracked_results.
    """
    vehicles = tracking.get("vehicles") or []
    recs = []
    for plate in tracking.get("plates") or []:
        tid = plate["track_id"]
        cached = plate_cache.get(tid)
        text, ocr_conf, is_valid = "", 0.0, False
        if cached:
            text = (cached.get("text") or "").strip()
            ocr_conf = float(cached.get("ocr_conf") or 0.0)
            is_valid = bool(cached.get("is_valid", False))
        v_tid = match_vehicle_for_plate(plate["bbox"], vehicles)
        recs.append({
            "bbox": plate["bbox"],
            "text": text,
            "ocr_conf": ocr_conf,
            "track_id": tid,
            "is_valid": is_valid,
            "vehicle_track_id": v_tid,
            "det_conf": plate["conf"],
        })
    return recs


_COLOR_CACHE: dict = {}

def _bgr_color_for_track(track_id):
    """Distinct BGR color per track id (vehicles + plates stay matched)."""
    if track_id is None:
        return (160, 160, 160)
    if track_id in _COLOR_CACHE:
        return _COLOR_CACHE[track_id]
    hue = int((abs(track_id) * 37) % 180)
    hsv = np.uint8([[[hue, 200, 235]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    color = tuple(int(x) for x in bgr)
    _COLOR_CACHE[track_id] = color
    return color


# Tracked Plate Cache (for real-time video)

class TrackedPlateCache:
    """
    Caches OCR results per YOLO track ID.
    Only re-OCR a plate when:
      - It has never been read, OR
      - A sharper crop appears (higher blur score)
    """

    def __init__(self, expire_seconds=5.0):
        self.expire_seconds = expire_seconds
        self._cache: dict[int, dict] = {}
        self._lock = threading.Lock()

    def needs_ocr(self, track_id, blur_score):
        """Return True if this track needs (re-)OCR."""
        with self._lock:
            if track_id not in self._cache:
                return True
            entry = self._cache[track_id]
            if blur_score > entry.get("blur_score", 0) * 1.5:
                return True
            if not entry.get("text"):
                return True
            return False

    def store(self, track_id, text, ocr_conf, raw_text, is_valid, blur_score):
        """Store or update OCR result for a track."""
        with self._lock:
            self._cache[track_id] = {
                "text": text,
                "ocr_conf": ocr_conf,
                "raw_text": raw_text,
                "is_valid": is_valid,
                "blur_score": blur_score,
                "last_seen": time.monotonic(),
            }

    def get(self, track_id):
        """Get cached result for a track, or None (copy safe for cross-thread read)."""
        with self._lock:
            entry = self._cache.get(track_id)
            if not entry:
                return None
            entry["last_seen"] = time.monotonic()
            return dict(entry)

    def mark_seen(self, track_id):
        """Update last_seen without changing OCR data."""
        with self._lock:
            if track_id in self._cache:
                self._cache[track_id]["last_seen"] = time.monotonic()

    def cleanup(self):
        """Remove expired entries."""
        with self._lock:
            now = time.monotonic()
            expired = [tid for tid, e in self._cache.items()
                       if now - e["last_seen"] > self.expire_seconds]
            for tid in expired:
                del self._cache[tid]


# ─────────────────────────────────────────────
# Real-time pipeline shared defaults
# ─────────────────────────────────────────────

OCR_QUEUE_MAXSIZE = 16
SMOOTH_WINDOW = 28
SMOOTH_MIN_VOTES = 3
SMOOTH_STALE_FRAMES = 42


def ocr_worker(recognizer, plate_cache, in_q, stop_event):
    """Background thread: drain OCR queue and populate plate_cache."""
    while not stop_event.is_set():
        try:
            item = in_q.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            break
        track_id, crop, blur_score = item
        text, conf, raw, valid = recognizer.ocr_plate_crop(crop)
        plate_cache.store(track_id, text, conf, raw, valid, blur_score)


# ─────────────────────────────────────────────
# Model Discovery Helper
# ─────────────────────────────────────────────

def find_best_model():
    """Search weights/, repo root, then Ultralytics runs for best.pt."""
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
    End-to-end Vietnamese license plate recognition.
    Combines YOLOv8 (detection) and PaddleOCR (character reading).
    """

    def __init__(self, yolo_model_path, confidence_threshold=0.3,
                 plate_class_id=PLATE_CLASS_ID, use_gpu=False):
        """
        Args:
            yolo_model_path:      path to .pt weights file
            confidence_threshold: minimum detection confidence (0–1)
            plate_class_id:       class index for 'plate' in the dataset
            use_gpu:              True to use GPU for PaddleOCR
        """
        print("Loading YOLOv8 model...")
        self.yolo = YOLO(yolo_model_path)

        print("Loading PaddleOCR...")
        self.ocr = PaddleOCR(
            lang="en",
            device="gpu" if use_gpu else "cpu",
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        self.conf_threshold = confidence_threshold
        self.plate_class_id = plate_class_id
        self._vehicle_classes = {0, 1}   # car, motorcycle
        self._tracker_yaml = _resolve_tracker_yaml()
        print("[DONE] Pipeline is ready!")

    def detect_plates(self, frame):
        """
        Run YOLOv8 on a frame and return plate bounding boxes.

        Returns:
            list of (x1, y1, x2, y2, confidence)
        """
        results = self.yolo(frame, conf=self.conf_threshold, verbose=False)
        plates = []
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) != self.plate_class_id:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                plates.append([x1, y1, x2, y2, conf])
                
        # Merge overlapping boxes (fixes YOLO splitting a large close-up plate into pieces)
        merged = []
        while plates:
            base = plates.pop(0)
            base_x1, base_y1, base_x2, base_y2, base_conf = base
            
            i = 0
            while i < len(plates):
                x1, y1, x2, y2, conf = plates[i]
                # Calculate Intersection
                ix1, iy1 = max(base_x1, x1), max(base_y1, y1)
                ix2, iy2 = min(base_x2, x2), min(base_y2, y2)
                inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                
                area1 = (base_x2 - base_x1) * (base_y2 - base_y1)
                area2 = (x2 - x1) * (y2 - y1)
                
                # If overlap is > 30% of the smaller box, merge them
                if inter_area > 0.3 * min(area1, area2):
                    base_x1, base_y1 = min(base_x1, x1), min(base_y1, y1)
                    base_x2, base_y2 = max(base_x2, x2), max(base_y2, y2)
                    base_conf = max(base_conf, conf)
                    plates.pop(i)
                else:
                    i += 1
            merged.append((base_x1, base_y1, base_x2, base_y2, base_conf))
            
        return merged

    def track_frame(self, frame, imgsz=480):
        """
        Run YOLO tracking on a frame. Returns all detections with
        persistent track IDs (vehicles + plates). FAST (~30-70ms).

        Returns:
            dict with:
                'vehicles': list of {bbox, class_name, conf, track_id}
                'plates':   list of {bbox, conf, track_id, crop, blur_score}
        """
        results = self.yolo.track(
            frame, conf=self.conf_threshold, persist=True,
            verbose=False, imgsz=imgsz,
            tracker=self._tracker_yaml,
        )

        h, w = frame.shape[:2]
        vehicles = []
        plates = []

        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue
            for box in result.boxes:
                if box.id is None:
                    continue
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                track_id = int(box.id[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                if cls in self._vehicle_classes:
                    vehicles.append({
                        "bbox": (x1, y1, x2, y2),
                        "class_name": self.yolo.names[cls],
                        "conf": conf,
                        "track_id": track_id,
                    })
                elif cls == self.plate_class_id:
                    pad_x = max(8, int((x2 - x1) * 0.10))
                    pad_y = max(5, int((y2 - y1) * 0.10))
                    crop = frame[max(0, y1 - pad_y): min(h, y2 + pad_y),
                                 max(0, x1 - pad_x): min(w, x2 + pad_x)]
                    blur = estimate_blur(crop) if crop.size > 0 else 0.0
                    plates.append({
                        "bbox": (x1, y1, x2, y2),
                        "conf": conf,
                        "track_id": track_id,
                        "crop": crop,
                        "blur_score": blur,
                    })

        return {"vehicles": vehicles, "plates": plates}

    def ocr_plate_crop(self, crop):
        """
        Run OCR on a single plate crop. Returns (cleaned_text, conf, raw, valid).
        This is the SLOW part (~100-300ms per plate).
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
        Detect plates and read their characters (single-shot, for images).

        Returns:
            list of dicts with keys:
                bbox, detection_conf, raw_text, text, ocr_conf,
                is_valid, plate_crop
        """
        recognitions = []
        h, w = frame.shape[:2]

        for (x1, y1, x2, y2, det_conf) in self.detect_plates(frame):
            pad = 5
            crop = frame[max(0, y1 - pad): min(h, y2 + pad),
                         max(0, x1 - pad): min(w, x2 + pad)]
            if crop.size == 0:
                continue

            raw_text, ocr_conf = read_plate_text(self.ocr, crop)

            # Skip results with very low OCR confidence
            if ocr_conf < MIN_OCR_CONFIDENCE:
                continue

            cleaned = clean_plate_text(raw_text)
            valid = is_valid_vietnam_plate(cleaned)

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
        Draw bounding boxes and plate text on a copy of the frame.

        Returns:
            annotated image (numpy array, BGR)
        """
        output = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        for rec in recognitions:
            x1, y1, x2, y2 = rec["bbox"]
            text = rec["text"]
            ocr_conf = rec["ocr_conf"]
            color = (0, 255, 0) if rec.get("is_valid", True) else (0, 215, 255)

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Skip label if text is empty (plate detected but not yet stabilized)
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

    def draw_tracked_results(self, frame, tracking, smoothed_plates):
        """
        Draw vehicles and plates with temporally smoothed OCR (PlateTracker output).

        Args:
            tracking:       dict from track_frame (vehicles + plates)
            smoothed_plates: list from PlateTracker.update(prepare_plate_recognitions(...));
                             same length and order as tracking["plates"].

        Returns:
            annotated image (numpy array, BGR)
        """
        output = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        font_scale_small = 0.55

        # Vehicles: color by track id + short label
        for v in tracking.get("vehicles") or []:
            x1, y1, x2, y2 = v["bbox"]
            tid = v["track_id"]
            color = _bgr_color_for_track(tid)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            vlabel = f"#{tid} {v['class_name']}"
            cv2.putText(output, vlabel, (x1, max(y1 - 6, 18)),
                        font, font_scale_small, color, 1, cv2.LINE_AA)

        plates = tracking.get("plates") or []
        for plate, row in zip(plates, smoothed_plates):
            x1, y1, x2, y2 = plate["bbox"]
            text = (row.get("text") or "").strip()
            ocr_conf = float(row.get("ocr_conf") or 0.0)
            valid = bool(row.get("is_valid", False))
            v_tid = row.get("vehicle_track_id")
            pl_tid = plate["track_id"]
            seed = v_tid if v_tid is not None else pl_tid
            box_color = _bgr_color_for_track(seed)

            if text:
                ok_color = (0, 220, 80) if valid else (0, 180, 255)
                label = f"{text} ({ocr_conf:.0%})"
                label_rgb = ok_color
            else:
                label = "..."
                label_rgb = (200, 200, 200)

            cv2.rectangle(output, (x1, y1), (x2, y2), box_color, 2)
            font_scale = 0.62
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            label_y = max(y1 - 10, th + 5)
            cv2.rectangle(output,
                          (x1, label_y - th - 4),
                          (x1 + tw + 4, label_y + 3),
                          box_color, -1)
            cv2.putText(output, label, (x1 + 2, label_y),
                        font, font_scale, label_rgb, thickness, cv2.LINE_AA)

        return output
