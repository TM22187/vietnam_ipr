"""Dịch vụ chạy nhận dạng ngoài UI thread."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from app_events import AppEvents
from app_runtime import AppSettings, verify_model
from lpr_pipeline import LicensePlateRecognizer, find_best_model


logger = logging.getLogger(__name__)
MAX_IMAGE_BYTES = 100 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000


def read_image(path: str) -> np.ndarray:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError("File ảnh không còn tồn tại")
    if image_path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError("Ảnh quá lớn (giới hạn 100 MB)")
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError) as exc:
        raise ValueError("Không đọc được file ảnh đã chọn") from exc
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Định dạng ảnh không được hỗ trợ")
    if frame.shape[0] * frame.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError("Độ phân giải ảnh quá lớn (giới hạn 40 megapixel)")
    return frame


class LatestCamera:
    """Luôn cung cấp frame mới nhất và đóng reader thread có kiểm soát."""

    def __init__(self, index: int = 0, warmup_timeout: float = 5.0):
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(index, backend)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = cv2.VideoCapture(index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Không thể mở camera {index}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._running = threading.Event()
        self._running.set()
        self._thread = threading.Thread(
            target=self._read_loop, name="camera-reader", daemon=True,
        )
        self._thread.start()

        deadline = time.monotonic() + warmup_timeout
        while self.read() is None and time.monotonic() < deadline:
            time.sleep(0.03)
        if self.read() is None:
            self.release()
            raise RuntimeError("Camera đã mở nhưng không trả về hình ảnh")

    def _read_loop(self) -> None:
        failures = 0
        while self._running.is_set():
            ok, frame = self.capture.read()
            if ok and frame is not None:
                failures = 0
                with self._lock:
                    self._frame = frame
            else:
                failures += 1
                if failures == 30:
                    logger.warning("Camera mất frame liên tục")
                time.sleep(0.03)

    def read(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def release(self) -> None:
        self._running.clear()
        self.capture.release()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.5)


class RecognitionWorker:
    def __init__(self, events: AppEvents, stop_event: threading.Event, settings: AppSettings):
        self.events = events
        self.stop_event = stop_event
        self.settings = settings
        self.recognizer: LicensePlateRecognizer | None = None

    def _status(self, text: str) -> None:
        self.events.publish("status", text)

    def _load_model(self) -> LicensePlateRecognizer:
        if self.recognizer is None:
            self._status("Đang kiểm tra và khởi tạo mô hình…")
            model = find_best_model()
            if not model:
                raise FileNotFoundError("Thiếu model models/best_vietnam_lpr.onnx")
            verify_model(model)
            started = time.perf_counter()
            self.recognizer = LicensePlateRecognizer(
                model, confidence_threshold=self.settings.detector_confidence,
            )
            logger.info("Model sẵn sàng sau %.2f giây", time.perf_counter() - started)
        self.recognizer.reset_stream()
        return self.recognizer

    def run(self, source_type: str, source: str | int, source_name: str) -> None:
        logger.info("Bắt đầu nguồn type=%s name=%s", source_type, source_name)
        try:
            recognizer = self._load_model()
            if self.stop_event.is_set():
                return
            if source_type == "image":
                self._run_image(recognizer, str(source), source_type, source_name)
            elif source_type == "video":
                self._run_video(recognizer, str(source), source_type, source_name)
            elif source_type == "camera":
                self._run_camera(recognizer, int(source), source_type, source_name)
            else:
                raise ValueError(f"Nguồn không hợp lệ: {source_type}")
        except Exception as exc:
            logger.exception("Lỗi xử lý nguồn")
            self.events.publish("error", str(exc))
        finally:
            logger.info("Kết thúc nguồn type=%s name=%s", source_type, source_name)
            self.events.publish("done")

    def _publish_plate(self, result: dict, source_type: str, source_name: str) -> None:
        enriched = dict(result)
        enriched["source_type"] = source_type
        enriched["source_name"] = source_name
        self.events.publish("plate", enriched)

    def _run_image(
        self, recognizer: LicensePlateRecognizer, path: str, source_type: str, source_name: str,
    ) -> None:
        frame = read_image(path)
        self._status("Đang nhận dạng ảnh…")
        results = recognizer.recognize(frame)
        self.events.publish("frame", recognizer.draw_results(frame, results))
        for result in results:
            if result.get("text"):
                self._publish_plate(result, source_type, source_name)
        detected = sum(1 for result in results if result.get("text"))
        self._status(f"Hoàn tất · đọc được {detected} biển số")

    def _run_video(
        self, recognizer: LicensePlateRecognizer, path: str, source_type: str, source_name: str,
    ) -> None:
        video = Path(path)
        if not video.is_file():
            raise FileNotFoundError("File video không còn tồn tại")
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError("Không mở được file video đã chọn")
        total = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        stride = max(1, round(fps / self.settings.video_target_fps))
        frame_index = 0
        self._status("Đang phân tích video…")
        try:
            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1
                if frame_index % stride:
                    continue
                output, _, new_events = recognizer.process_stream_frame(frame)
                self.events.publish("frame", output)
                for event in new_events:
                    self._publish_plate(event, source_type, source_name)
                if total and frame_index % max(stride * 15, 1) == 0:
                    self._status(f"Đang phân tích video · {min(100, frame_index * 100 // total)}%")
        finally:
            capture.release()
        if not self.stop_event.is_set():
            self._status("Đã phân tích xong video")

    def _run_camera(
        self, recognizer: LicensePlateRecognizer, index: int, source_type: str, source_name: str,
    ) -> None:
        camera = LatestCamera(index)
        self._status("Camera đang hoạt động")
        try:
            while not self.stop_event.is_set():
                frame = camera.read()
                if frame is None:
                    time.sleep(0.02)
                    continue
                output, _, new_events = recognizer.process_stream_frame(frame)
                self.events.publish("frame", output)
                for event in new_events:
                    self._publish_plate(event, source_type, source_name)
        finally:
            camera.release()
