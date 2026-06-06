"""
Nhận Dạng Biển Số Xe Việt Nam - CHẾ ĐỘ SNAPSHOT & LỊCH SỬ
==========================================================
Kiến trúc:
    1. ThreadedCamera: Đọc camera ngầm siêu mượt, loại bỏ delay của OpenCV.
       Tự kết nối lại khi camera mất kết nối.
    2. YOLOv8: Chạy liên tục để theo dõi khung biển số.
    3. Snapshot OCR: Chỉ chụp và đọc ký tự ĐÚNG 1 LẦN khi xe tiến vào đủ gần (width > 100px).
    4. Lịch sử (Deque): Lưu và hiển thị 5 biển số hợp lệ gần nhất.
    5. ROI: Chỉ detect biển số trong vùng config/roi.yaml (nếu có).
"""

from pathlib import Path
import sys
import os
import time
import argparse
import logging
import cv2
import threading
from collections import deque

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lpr_pipeline import (
    LicensePlateRecognizer,
    find_best_model,
    PLATE_CLASS_ID,
)

logger = logging.getLogger("lpr.webcam")
CAPTURE_DIR = ROOT / "captures"

# ─── Đọc cấu hình ROI ──────────────────────────────────────

def load_roi_from_yaml(path=None):
    """Tải ROI từ config/roi.yaml. Trả về tuple (x1,y1,x2,y2) hoặc None."""
    roi_path = path or (ROOT / "config" / "roi.yaml")
    if not roi_path.exists():
        logger.info("Không tìm thấy file cấu hình ROI — phát hiện toàn khung hình")
        return None

    try:
        import yaml
        with open(roi_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except ImportError:
        # Fallback: đọc YAML thủ công (chỉ cần key: value đơn giản)
        cfg = {}
        for line in roi_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, val = line.split(":", 1)
                key, val = key.strip(), val.strip()
                # Thử parse thành số, nếu không thì giữ nguyên chuỗi
                try:
                    cfg[key] = float(val)
                except ValueError:
                    cfg[key] = val

    if not cfg.get("enabled", True) or str(cfg.get("enabled", "true")).lower() == "false":
        logger.info("Tìm thấy cấu hình ROI nhưng đang tắt")
        return None

    try:
        roi = (float(cfg["x1"]), float(cfg["y1"]),
               float(cfg["x2"]), float(cfg["y2"]))
        logger.info(f"ROI đã tải: x=[{roi[0]:.0%}–{roi[2]:.0%}], y=[{roi[1]:.0%}–{roi[3]:.0%}]")
        return roi
    except (KeyError, ValueError) as e:
        logger.warning(f"Cấu hình ROI không hợp lệ: {e} — phát hiện toàn khung hình")
        return None


# ─── Camera chạy trên luồng riêng, tự kết nối lại ─────────

class ThreadedCamera:
    """Đọc camera ở một luồng ngầm để luôn lấy được khung hình mới nhất (Real-time 100%)
       Tự kết nối lại khi camera mất kết nối."""

    MAX_RETRIES = 5
    RETRY_DELAY = 2.0  # giây

    def __init__(self, src=0):
        self.src = src
        self.cap = None
        self.ret = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self._consecutive_failures = 0
        self._open_camera()
        threading.Thread(target=self._update, daemon=True).start()

    def _open_camera(self):
        """Mở (hoặc mở lại) camera."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        ret, frame = self.cap.read()
        with self.lock:
            self.ret, self.frame = ret, frame

        if ret:
            logger.info(f"Camera {self.src} đã mở thành công")
            self._consecutive_failures = 0
        else:
            logger.warning(f"Camera {self.src} đã mở nhưng chưa có frame")

    def _update(self):
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                self._reconnect()
                continue

            ret, frame = self.cap.read()

            if ret:
                with self.lock:
                    self.ret, self.frame = ret, frame
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1
                if self._consecutive_failures > 30:
                    logger.warning(f"Camera {self.src}: {self._consecutive_failures} lần đọc thất bại liên tiếp — đang kết nối lại")
                    self._reconnect()

            time.sleep(0.01)  # Nhường một chút CPU cho luồng chính

    def _reconnect(self):
        """Thử kết nối lại camera."""
        for attempt in range(1, self.MAX_RETRIES + 1):
            if self.stopped:
                return
            logger.info(f"Đang kết nối lại camera, lần thử {attempt}/{self.MAX_RETRIES}...")
            time.sleep(self.RETRY_DELAY)
            self._open_camera()
            if self.ret:
                return
        logger.error(f"Camera {self.src}: tất cả {self.MAX_RETRIES} lần kết nối lại đều thất bại")

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def release(self):
        self.stopped = True
        if self.cap:
            self.cap.release()


# ─── Vòng lặp chính - Chế độ Snapshot ───────────────────────

def run_webcam_snapshot(recognizer, camera_index=0, use_gpu=False):
    cap = ThreadedCamera(camera_index)
    time.sleep(1.0)  # Đợi camera khởi động lên hình

    device = "cuda" if use_gpu else "cpu"
    recognizer.yolo.to(device)
    logger.info(f"YOLO đang chạy trên: {device.upper()}")
    logger.info(f"Hệ thống đang chạy ở chế độ: SNAPSHOT MODE (Trạm Thu Phí)")

    ocr_cache = {}
    fps_times = []
    current_fps = 0.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_count = 0

    # Hàng đợi lưu tối đa 5 biển số gần nhất
    recent_plates = deque(maxlen=5)

    # Ngưỡng kích hoạt chụp ảnh: chiều rộng biển số (pixel)
    SNAPSHOT_WIDTH_THRESHOLD = 100

    # ── Cache TTL ── Dọn entry cũ hơn 60s để tránh memory leak
    OCR_CACHE_TTL = 60.0
    CACHE_CLEANUP_INTERVAL = 100  # mỗi 100 frame

    # BƯỚC 1: KHỞI TẠO SỔ TAY COOLDOWN (Cấm đọc lại biển trùng trong 10 giây)
    plate_cooldown = {}
    COOLDOWN_SECONDS = 10.0

    while True:
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            now = time.time()
            frame_count += 1

            # Tính FPS
            fps_times.append(now)
            if len(fps_times) > 30:
                fps_times.pop(0)
            if len(fps_times) >= 2:
                current_fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0])

            h, w = frame.shape[:2]
            output = frame.copy()

            # ── Dọn ocr_cache và cooldown định kỳ ──
            if frame_count % CACHE_CLEANUP_INTERVAL == 0:
                before = len(ocr_cache)
                ocr_cache = {
                    k: v for k, v in ocr_cache.items()
                    if now - v.get("timestamp", 0) < OCR_CACHE_TTL
                }
                after = len(ocr_cache)
                if before != after:
                    logger.debug(f"Dọn cache: {before} → {after} entries")

                # BƯỚC 2: DỌN DẸP SỔ TAY COOLDOWN QUÁ HẠN
                plate_cooldown = {k: v for k, v in plate_cooldown.items() if now - v < COOLDOWN_SECONDS}

            # ── Vẽ overlay vùng ROI ──
            roi_px = recognizer.get_roi_pixels(w, h)
            if roi_px:
                rx1, ry1, rx2, ry2 = roi_px
                overlay = output.copy()
                cv2.rectangle(overlay, (0, 0), (w, ry1), (0, 0, 0), -1)          # trên
                cv2.rectangle(overlay, (0, ry2), (w, h), (0, 0, 0), -1)          # dưới
                cv2.rectangle(overlay, (0, ry1), (rx1, ry2), (0, 0, 0), -1)      # trái
                cv2.rectangle(overlay, (rx2, ry1), (w, ry2), (0, 0, 0), -1)      # phải
                output = cv2.addWeighted(output, 0.7, overlay, 0.3, 0)
                cv2.rectangle(output, (rx1, ry1), (rx2, ry2), (0, 255, 255), 1)  # viền ROI
                cv2.putText(output, "ROI", (rx1 + 5, ry1 + 18), font, 0.5, (0, 255, 255), 1)

            # ---------------------------------------------------------
            # 1. Phát hiện và theo dõi biển số bằng YOLO
            # ---------------------------------------------------------
            results = recognizer.yolo.track(
                frame, conf=recognizer.conf_threshold, persist=True,
                verbose=False, imgsz=640, device=device, half=use_gpu
            )

            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                for box in results[0].boxes:
                    if int(box.cls[0]) != PLATE_CLASS_ID:
                        continue

                    track_id = int(box.id[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    # ── Lọc ROI ──
                    if not recognizer._is_inside_roi(x1, y1, x2, y2, w, h):
                        continue

                    box_width = x2 - x1

                    cached = ocr_cache.get(track_id, {})
                    text = cached.get("text", "")
                    ocr_conf = cached.get("ocr_conf", 0.0)
                    is_valid = cached.get("is_valid", False)

                    # ---------------------------------------------------------
                    # 2. LOGIC CHỤP ẢNH (SNAPSHOT)
                    # ---------------------------------------------------------
                    if not is_valid and box_width > SNAPSHOT_WIDTH_THRESHOLD:
                        cv2.putText(output, ">>> DANG DOC BIEN SO...", (x1, max(20, y1 - 30)), font, 0.7, (0, 0, 255), 2)
                        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.imshow("Nhan Dang Bien So - Che Do Chup Anh", output)
                        cv2.waitKey(1)

                        pad = 20
                        crop = frame[max(0, y1-pad): min(h, y2+pad), max(0, x1-pad): min(w, x2+pad)]

                        if crop.size > 0:
                            cleaned, conf, raw, valid = recognizer.ocr_plate_crop(crop)

                            if valid or conf > 0.4:
                                ocr_cache[track_id] = {
                                    "text": cleaned,
                                    "ocr_conf": conf,
                                    "is_valid": valid,
                                    "timestamp": time.time(),
                                }
                                text, ocr_conf, is_valid = cleaned, conf, valid

                                # BƯỚC 3: KIỂM TRA COOLDOWN TRƯỚC KHI GHI NHẬN LỊCH SỬ
                                # Nới lỏng: Chấp nhận biển hợp lệ hoặc đọc được >= 5 ký tự với độ tự tin tốt
                                if valid or (len(text) >= 5 and ocr_conf > 0.6):
                                    last_seen = plate_cooldown.get(text, 0)

                                    # Nếu đã qua 10 giây kể từ lần cuối đọc biển này → Cho phép ghi nhận
                                    if now - last_seen > COOLDOWN_SECONDS:
                                        logger.info(f">>> [GHI NHAN] {text} ({ocr_conf:.0%})")
                                        plate_info = f"{text} ({ocr_conf:.0%})"

                                        recent_plates.appendleft(plate_info)
                                        plate_cooldown[text] = now  # Cập nhật thời điểm vừa đọc

                                        # Khóa track_id này lại để không OCR thêm nữa
                                        ocr_cache[track_id]["is_valid"] = True
                                        is_valid = True

                    # ---------------------------------------------------------
                    # 3. Vẽ khung hiển thị trực tiếp theo xe
                    # ---------------------------------------------------------
                    color = (0, 255, 0) if is_valid else (0, 215, 255)
                    cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

                    if text:
                        label = f"{text} ({ocr_conf:.0%})"
                    else:
                        label = "Tien lai gan hon..." if box_width <= SNAPSHOT_WIDTH_THRESHOLD else "..."

                    cv2.putText(output, label, (x1, max(20, y1 - 10)), font, 0.7, color, 2, cv2.LINE_AA)

            # ---------------------------------------------------------
            # 4. Vẽ Bảng Lịch sử 5 Biển Số
            # ---------------------------------------------------------
            panel_x = w - 260
            cv2.putText(output, "5 BIEN SO GAN NHAT:", (panel_x, 30), font, 0.6, (0, 255, 255), 2)  # Không dấu vì OpenCV không hỗ trợ font Unicode

            for i, p_info in enumerate(recent_plates):
                y_pos = 60 + (i * 30)
                cv2.putText(output, f"{i+1}. {p_info}", (panel_x, y_pos), font, 0.6, (0, 255, 0), 2)

            cv2.putText(output, f"FPS: {current_fps:.1f}", (10, 30), font, 0.7, (0, 255, 255), 2)
            cv2.putText(output, f"Cache: {len(ocr_cache)}", (10, 55), font, 0.5, (180, 180, 180), 1)

            cv2.imshow("Nhan Dang Bien So - Che Do Chup Anh", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(CAPTURE_DIR / f"capture_{int(time.time())}.jpg"), output)
                logger.info("Đã lưu ảnh!")

        except Exception as e:
            logger.error(f"Lỗi trong vòng lặp chính: {e}", exc_info=True)
            time.sleep(0.1)
            continue

    cap.release()
    cv2.destroyAllWindows()


# Alias tương thích ngược với demo.py
def run_webcam_demo(recognizer, camera_index=0, use_gpu=False):
    """Hàm alias để tương thích ngược, được gọi từ demo.py."""
    run_webcam_snapshot(recognizer, camera_index=camera_index, use_gpu=use_gpu)


def main():
    # ── Cấu hình logging ──
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Nhận dạng biển số xe Việt Nam thời gian thực (Snapshot Mode)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model",  default=None)
    parser.add_argument("--conf",   type=float, default=0.25)
    parser.add_argument("--gpu",    action="store_true")
    parser.add_argument("--no-roi", action="store_true",
                        help="Tắt ROI dù có file config/roi.yaml")
    args = parser.parse_args()

    model_path = args.model or find_best_model()
    if not model_path:
        logger.error("Không tìm thấy file weights YOLO!")
        return

    # Tải ROI
    roi = None if args.no_roi else load_roi_from_yaml()

    recognizer = LicensePlateRecognizer(
        model_path, confidence_threshold=args.conf, roi=roi)
    run_webcam_snapshot(recognizer, camera_index=args.camera, use_gpu=args.gpu)

if __name__ == "__main__":
    main()