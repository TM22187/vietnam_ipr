# Vietnam LPR Edge C++

Đây là runtime thứ hai dành cho phần cứng. Nó chạy headless 24/7, không có giao diện,
không chọn ROI và không cần Python khi triển khai. App Windows ở thư mục gốc vẫn được
giữ nguyên để demo, kiểm tra camera và xem lịch sử.

## Kiến trúc

Luồng xử lý được rút gọn cho bài toán biển số:

`camera/video -> YOLOv8 ONNX -> crop biển -> PP-OCRv6 ONNX -> chuẩn hóa -> JSON event`

- ONNX Runtime C++ chạy detector và recognizer trên CPU.
- Không chạy text detector thứ hai vì YOLO đã cung cấp crop biển. Biển vuông được tách
  hai dòng bằng phép chiếu ảnh; biển dài được nhận dạng trực tiếp.
- Track theo IoU giữ kết quả OCR trong 1,5 giây và chỉ thử lại sau 0,9 giây nếu chưa hợp
  lệ. Điều này giảm tải CPU rõ rệt trên video.
- Camera lỗi được mở lại; `SIGINT`/`SIGTERM` dừng sạch; heartbeat và sự kiện được ghi
  dạng JSON Lines để PLC, relay, MQTT bridge hoặc backend đọc độc lập.

Baseline phù hợp với máy Linux/Windows x86-64 hoặc ARM64 có tối thiểu khoảng 1 GB RAM.
Đây không phải firmware cho ESP32/STM32: vi điều khiển loại đó cần model nhỏ hơn và
pipeline khác. Jetson có thể thay execution provider CPU bằng TensorRT sau khi chốt
đúng bo mạch mà không đổi schema sự kiện.

## Build Linux

Yêu cầu: CMake 3.20+, compiler C++17, OpenCV 4 development package và ONNX Runtime
C/C++ SDK đúng kiến trúc máy.

```bash
sudo apt install build-essential cmake libopencv-dev
export ONNXRUNTIME_ROOT=/opt/onnxruntime
./scripts/build-linux.sh
```

Nếu build trực tiếp:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT=/opt/onnxruntime
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Build Windows

Dùng MSVC x64, OpenCV C++ SDK và ONNX Runtime C/C++ SDK:

```powershell
$env:ONNXRUNTIME_ROOT = "C:\sdk\onnxruntime"
$env:OpenCV_DIR = "C:\sdk\opencv\build\x64\vc16\lib"
.\scripts\build-windows.ps1
```

CMake tự chép `onnxruntime.dll`, OpenCV runtime và video backend cạnh file `.exe` để
Windows không nạp nhầm bản ONNX Runtime cũ trong `System32`.

## Chạy

Chương trình tự tìm model khi chạy từ repository hoặc sau `cmake --install`.

```bash
# Camera /dev/video0
./build/vlpr_edge --source 0 --threads 2 \
  --event-log /var/lib/vietnam-lpr-edge/events.jsonl

# RTSP
./build/vlpr_edge --source rtsp://user:password@camera/stream1 --frame-stride 2

# Kiểm tra model mà chưa mở camera
./build/vlpr_edge --dry-run

# Smoke test và benchmark trên ảnh
./build/vlpr_edge --self-test ../xe.jpg
./build/vlpr_edge --benchmark ../xe.jpg --iterations 30
```

Một event được xuất trên `stdout` và tùy chọn file JSONL:

```json
{"ts":"2026-06-30T05:00:00.000Z","type":"plate","source":"0","plate":"51A69172","valid":true,"detection_confidence":0.94,"ocr_confidence":0.84,"bbox":[306,396,440,458],"detector_ms":136.2,"total_ms":228.4}
```

Không điều khiển GPIO trực tiếp trong tiến trình AI. Một service nhỏ nên đọc event
`type=plate`, áp dụng whitelist/quy tắc an toàn rồi mới kích relay. Cách tách này tránh
việc lỗi camera hoặc AI giữ cổng ở trạng thái nguy hiểm.

## Cài service Linux

```bash
sudo cmake --install build --prefix /usr
sudo useradd --system --no-create-home --groups video vlpr || true
sudo install -d -o vlpr -g video /var/lib/vietnam-lpr-edge
sudo install -m 0644 systemd/vietnam-lpr-edge.service \
  /etc/systemd/system/vietnam-lpr-edge.service
sudo systemctl daemon-reload
sudo systemctl enable --now vietnam-lpr-edge
journalctl -u vietnam-lpr-edge -f
```

Sửa `User`, `Group`, `--source`, số thread và đường dẫn event trong unit theo bo mạch.
Với camera USB, user chạy service phải thuộc group `video`.

## Kiểm tra nhanh không cần OpenCV/ONNX Runtime

Phần chuẩn hóa biển và JSON có thể compile độc lập trên máy build:

```bash
cmake -S . -B build-core -DVLPR_BUILD_RUNTIME=OFF -DVLPR_BUILD_TESTS=ON
cmake --build build-core
ctest --test-dir build-core --output-on-failure
```

Model và charset được khóa checksum tại `models/model_manifest.json`. Charset UTF-8
được trích từ metadata của model bằng `tools/export_charset.py`; không sửa tay file này.
