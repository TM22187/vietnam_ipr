# Vietnam LPR Desktop

Ứng dụng Windows nhận dạng biển số xe Việt Nam từ **ảnh, video hoặc camera**.
Toàn bộ inference chạy offline trên CPU. Giao diện không có ROI và người dùng
không cần cài Python.

## Kiến trúc runtime

- YOLOv8 ONNX phát hiện biển số; RapidOCR ONNX đọc ký tự.
- ONNX Runtime và OpenCV thực thi phần tính toán nặng bằng native C/C++.
- Worker nhận dạng tách khỏi UI thread; frame queue có back-pressure và chỉ giữ
  frame mới nhất để tránh tăng RAM khi camera chạy lâu.
- Camera reader có warm-up, timeout và shutdown có kiểm soát.
- Model được kiểm tra SHA-256 trước khi nạp; model sai hoặc hỏng sẽ bị từ chối.
- Lịch sử lưu bằng SQLite/WAL và có thể xuất CSV.
- Log xoay vòng và crash report nằm trong `%LOCALAPPDATA%\VietnamLPR\logs`.
- Chỉ cho phép một instance để tránh tranh camera và database.

## Dữ liệu vận hành

Ứng dụng không ghi dữ liệu vào thư mục cài đặt:

```text
%LOCALAPPDATA%\VietnamLPR\
├── config.json
├── data\recognitions.db
└── logs\vietnam-lpr.log
```

Gỡ ứng dụng không tự xóa lịch sử người dùng.

## Chạy từ mã nguồn

Yêu cầu Windows 10/11 64-bit và Python 3.12 hoặc 3.13 có Tcl/Tk.

```powershell
python -m venv .venv-app
.\.venv-app\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-app\Scripts\python.exe desktop_app.py
```

Model bắt buộc nằm tại `models/best_vietnam_lpr.onnx` và phải khớp checksum
trong `models/model_manifest.json`.

## Build có kiểm soát chất lượng

```powershell
.\build_app.ps1
```

Script build tự động:

1. kiểm tra checksum model;
2. sinh icon đa kích thước;
3. chạy toàn bộ unit test;
4. build PyInstaller;
5. smoke-test chính file `.exe` vừa build;
6. ghi `build-info.json`;
7. tạo installer Inno Setup nếu compiler có trên máy.

`requirements-lock.txt` khóa toàn bộ dependency của môi trường release
Windows x64/Python 3.13; `requirements.txt` chỉ dùng cho phát triển.

Kết quả:

- Portable: `dist\VietnamLPR\VietnamLPR.exe`
- Installer: `dist\installer\VietnamLPR-Setup-1.1.0.exe`

Installer cài theo tài khoản người dùng và không yêu cầu Administrator.

### Ký số bản phát hành

Build script hỗ trợ Authenticode khi CI/máy release cung cấp:

```powershell
$env:CODE_SIGN_CERT_SHA1 = "CERTIFICATE_THUMBPRINT"
$env:SIGNTOOL_PATH = "C:\path\to\signtool.exe"
.\build_app.ps1
```

Không có certificate thì app vẫn build được nhưng Windows SmartScreen có thể
hiện cảnh báo. Bản phát hành production nên luôn được ký số và timestamp.

## Kiểm thử thủ công

```powershell
.\.venv-app\Scripts\python.exe -m unittest discover -s tests -v
.\dist\VietnamLPR\VietnamLPR.exe --smoke-test
```

Workflow `.github/workflows/quality.yml` chạy test, package và smoke test trên
Windows cho mỗi push/PR. Thông tin license của dependency nằm trong
`THIRD_PARTY_NOTICES.md`.

Notebook train và cấu hình dataset được giữ lại để có thể huấn luyện model mới.
Các script CLI/OpenCV cũ đã bị loại bỏ vì trùng chức năng với app.
