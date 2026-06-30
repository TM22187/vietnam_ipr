# Vietnam LPR Desktop

Ứng dụng Windows nhận dạng biển số xe Việt Nam từ **ảnh, video hoặc camera**.
Toàn bộ xử lý chạy offline trên CPU; giao diện không có ROI và không cần mở
terminal.

## Điểm chính

- GUI desktop đơn giản, có preview và lịch sử biển số.
- YOLOv8 ONNX + RapidOCR ONNX; không còn PyTorch, Ultralytics, PaddlePaddle.
- Cache theo vị trí khi chạy video/camera để không OCR lặp lại mỗi frame.
- Camera luôn lấy frame mới nhất, hạn chế độ trễ.
- Có cấu hình PyInstaller và Inno Setup để tạo file cài đặt Windows.

## Chạy từ mã nguồn

Yêu cầu Windows 10/11 64-bit và Python 3.12 hoặc 3.13 có Tcl/Tk.

```powershell
python -m venv .venv-app
.\.venv-app\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-app\Scripts\python.exe desktop_app.py
```

Model bắt buộc nằm tại `models/best_vietnam_lpr.onnx`.

## Build app và installer

```powershell
.\build_app.ps1
```

Kết quả:

- App portable: `dist\VietnamLPR\VietnamLPR.exe`
- Installer (nếu máy có Inno Setup 6): `dist\installer\VietnamLPR-Setup-1.0.0.exe`

Installer cài theo tài khoản người dùng nên không yêu cầu quyền Administrator.

## Kiểm thử

```powershell
.\.venv-app\Scripts\python.exe -m unittest discover -s tests -v
```

Notebook train và cấu hình dataset được giữ lại để có thể huấn luyện model mới.
Các script CLI/OpenCV cũ đã được loại bỏ vì trùng chức năng với app.
