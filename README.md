# Vietnam LPR

YOLOv8 (detect + track xe / biển) + PaddleOCR đọc biển Việt Nam. Video/webcam có OCR nền và làm mượt chữ theo thời gian.

## Cấu trúc thư mục

```
vietnam_ipr/
├── lpr_pipeline.py       # Core: YOLO, OCR, xử lý ROI, vẽ kết quả
├── requirements.txt
├── README.md
├── config/
│   ├── bytetrack_lpr.yaml    # Tracker (traffic-friendly)
│   ├── data.yaml             # Train từ root: yolo detect train data=config/data.yaml ...
│   └── roi.yaml              # Vùng nhận dạng ROI (tùy chọn, gitignored)
├── scripts/
│   ├── run_webcam.py
│   ├── run_video.py
│   ├── test_image.py
│   └── set_roi.py            # Công cụ chọn vùng ROI tương tác
├── weights/
│   ├── best_vietnam_lpr.pt   # Put trained weights here (ưu tiên khi không truyền --model)
│   └── pretrained/           # yolov8s.pt / yolo26n.pt (train & notebook)
├── dataset/
│   └── data.yaml               # Train khi cwd là trong dataset/
├── notebooks/
│   └── vietnam_lpr_training.ipynb
├── captures/                   # Ảnh khi bấm 's' trong webcam/video (gitignored)
├── docs/                       # Tài liệu tùy chọn
└── runs/                       # Ultralytics training outputs (gitignored)
```

## Cài đặt

```bash
cd vietnam_ipr
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Cần file **`weights/best_vietnam_lpr.pt`** (hoặc truyền `--model`). PaddleOCR 3.x khớp API trong code (xem comment trong `requirements.txt`).

## Chạy inference

Luôn có thể chạy từ **thư mục gốc project** (`vietnam_ipr/`):

```bash
python scripts/run_webcam.py
python scripts/run_webcam.py --conf 0.25 --gpu

python scripts/run_video.py --video path/to/video.mp4
python scripts/run_video.py --video clip.mp4 --output out.mp4 --no-preview

python scripts/test_image.py --image photo.jpg
```

## Chạy desktop app

```bash
python app.py
```

App mở camera trực tiếp, hiển thị khung hình live và 5 biển số gần nhất. Có thể truyền tham số nếu cần:

```bash
python app.py --camera 1 --conf 0.25 --gpu
python app.py --no-roi
```

Để đóng gói thành app Windows:

```powershell
.\scripts\build_windows_app.ps1
```

File chạy nằm ở `dist\VietnamLPR\VietnamLPR.exe`.

## Cấu hình Vùng nhận dạng (ROI)

Để tối ưu hóa hiệu năng và tránh nhận dạng nhầm các biển số ở xa hoặc ngoài luồng xe, bạn có thể thiết lập vùng ROI (Region of Interest):

1. **Thiết lập ROI tương tác:**
   ```bash
   python scripts/set_roi.py
   # Hoặc dùng camera/video khác:
   python scripts/set_roi.py --camera 1
   python scripts/set_roi.py --video path/to/video.mp4
   ```
   *Cách dùng:* Kéo chuột chọn vùng cần nhận dạng, bấm **Enter/Space** để lưu vào `config/roi.yaml`, bấm **C** để chọn lại, hoặc **Q/Esc** để thoát.

2. **Chạy nhận dạng có ROI:**
   Mặc định `run_webcam.py` và `demo.py` sẽ tự động tải `config/roi.yaml` nếu tồn tại. Nếu bạn muốn tạm thời tắt vùng ROI và chạy trên toàn bộ khung hình, hãy truyền thêm tham số `--no-roi`:
   ```bash
   python scripts/run_webcam.py --no-roi
   ```

## Train YOLO

- **Notebook:** mở `notebooks/vietnam_lpr_training.ipynb`, chạy lần lượt các ô (ô cấu hình dataset sẽ neo `cwd` về root và tìm `dataset/`).
- **CLI từ root:** `yolo detect train data=config/data.yaml model=yolov8s.pt ...`

Sau train, notebook/script copy **`best.pt` → `weights/best_vietnam_lpr.pt`** để scripts inference tự tìm.

## Ghi chú

- `find_best_model()` tìm theo thứ tự: `weights/best_vietnam_lpr.pt`, root legacy, `runs/detect/**/best.pt`.
- Biến môi trường (tùy máy): `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` nếu Paddle treo khi check mạng lần đầu.
