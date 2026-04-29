# Hệ Thống Phát Hiện và Đếm Người
### Môn: Xử Lý Ảnh và Thị Giác Máy Tính (121036)
### Nhóm 12 — Trường ĐH Giao Thông Vận Tải TP. Hồ Chí Minh

---

## Giới thiệu

Dự án xây dựng hệ thống phát hiện và đếm người trong video theo thời gian thực, sử dụng pipeline kết hợp:
- **Tiền xử lý ảnh** (Gaussian Blur, Resize, CLAHE) — tăng chất lượng đầu vào (Ch.2: Xử lý ảnh)
- **Centroid Tracking** — theo dõi và gán ID từng người (Ch.4: Phân đoạn ảnh)
- **YOLOv8** — phát hiện người (Ch.5: Nhận dạng ảnh)

---

## Cấu trúc thư mục

```
XU_LY_ANH-NHOM_12/
├── data/
│   ├── images/
│   │   └── pic1.jpg
│   └── videos/
│       ├── video1.mp4
│       ├── video2.mp4
│       ├── video3.mp4
│       └── video4.mp4
├── models/
│   ├── yolov8n.pt          # Pretrained YOLOv8n
│   ├── best.pt             # Model tự train
│   └── last.pt
├── src/
│   ├── preprocessing.py    # Tiền xử lý ảnh (Ch.2)
│   ├── segmentation.py     # MOG2 + Morphology
│   ├── tracker.py          # Centroid Tracking (Ch.4)
│   ├── video_reader.py     # Đọc và xử lý video
│   ├── image_reader.py     # Đọc và xử lý ảnh
│   └── main.py             # Pipeline chính (entry point)
├── training/               # Kết quả huấn luyện model
│   └── train13/
├── README.md
└── requirements.txt
```

---

## Yêu cầu hệ thống

- Python 3.9 trở lên
- Webcam hoặc file video đầu vào
- GPU (khuyến nghị) hoặc CPU

---

## Cài đặt

**1. Clone hoặc giải nén project**
```bash
cd XU_LY_ANH-NHOM_12
```

**2. Cài đặt thư viện**
```bash
pip install -r requirements.txt
```

Nội dung `requirements.txt`:
```
opencv-python
ultralytics
scipy
numpy
```

---

## Chạy chương trình

**Chạy pipeline chính (video):**
```bash
python src/main.py
```

**Test tiền xử lý:**
```bash
python src/preprocessing.py
```

**Test đọc video:**
```bash
python src/video_reader.py
```

**Test đọc ảnh:**
```bash
python src/image_reader.py
```

---

## Cấu hình

Mở file `src/main.py` và chỉnh các tham số sau:

```python
VIDEO_PATH  = os.path.join(BASE_DIR, "data", "videos", "video3.mp4")  # Đường dẫn video
MODEL_PATH  = os.path.join(BASE_DIR, "models", "yolov8n.pt")           # Model sử dụng
LINE_Y      = 260    # Vị trí vạch đếm (pixel theo chiều dọc)
CONF        = 0.4    # Ngưỡng confidence của YOLO (0.0 - 1.0)
REAL_COUNT  = 10     # Số người thực tế (để tính MAE)
```

---

## Kết quả đầu ra

Sau khi chạy, chương trình sẽ:
- Hiển thị cửa sổ video theo thời gian thực với bounding box, ID và counting line
- Lưu video kết quả tại `output/result.avi`
- In báo cáo đánh giá ra console:

```
--- KẾT QUẢ ĐÁNH GIÁ ---
Số người thực tế : 10
Số người AI đếm  : 9
Sai số MAE       : 1
Độ chính xác     : 90.00%
Video kết quả    : .../output/result.avi
```

---

## Pipeline hoạt động

```
Input Video
    ↓
preprocessing.py   →   Gaussian Blur + Resize + CLAHE
    ↓
main.py (YOLOv8)   →   Phát hiện người → Bounding boxes
    ↓
tracker.py         →   Centroid Tracking → Gán ID từng người
    ↓
Counting Logic     →   Đếm người qua vạch LINE_Y
    ↓
Output Video + Evaluation Report
```

---

## Phím tắt

| Phím | Chức năng |
|------|-----------|
| `q`  | Thoát chương trình |

---

## Nhóm thực hiện — Nhóm 12

| STT | Họ tên                  | Mã SV | Phụ trách                 
|-----|--------------------------------------------------------------      
| 1   | Trần Hoài Nam           |       | `preprocessing.py`
| 2   | Đặng Thanh Duy          |       | `tracker.py`
| 3   | Nguyễn Anh Vinh         |       | `main.py`
| 4   | Phạm Trung Kiên         |       | `video_reader.py`, `image_reader.py`
| 5   | Đặng Phạm Văn Thành     |       | `segmentation.py`

---

## Tài liệu tham khảo

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com)
- [OpenCV Documentation](https://docs.opencv.org)
- Bewley et al. (2016). *Simple Online and Realtime Tracking (SORT)*