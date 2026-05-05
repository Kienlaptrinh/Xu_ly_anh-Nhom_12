"""
preprocessing.py — Tiền xử lý ảnh (Ch.2 + Ch.3)
=================================================
Chương 2 — Các kỹ thuật xử lý ảnh:
  - Gaussian Blur       : lọc tuyến tính giảm nhiễu
  - CLAHE               : cân bằng histogram cục bộ (toán tử điểm phi tuyến)
  - Resize              : biến đổi hình học
  - Canny Edge          : phát hiện cạnh (gradient)

Chương 3 — Phát hiện đặc trưng:
  - HOG Descriptor      : mô tả đặc trưng hướng gradient, dùng cho phát hiện người
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────
#  Ch.2 — Các hàm xử lý ảnh cơ bản
# ─────────────────────────────────────────────

def resize_keep_ratio(img, width=800):
    """
    Biến đổi hình học (Ch.2): resize giữ nguyên tỉ lệ khung hình.
    Dùng INTER_LINEAR vì cân bằng tốt giữa tốc độ và chất lượng
    khi phóng to/thu nhỏ nhẹ.
    """
    h, w = img.shape[:2]
    if w == width:
        return img
    scale = width / w
    new_h = int(h * scale)
    return cv2.resize(img, (width, new_h), interpolation=cv2.INTER_LINEAR)


def gaussian_blur(img, ksize=(5, 5), sigma=1.4):
    """
    Lọc tuyến tính (Ch.2): Gaussian Blur với kernel 5×5, σ=1.4.
    Theo lý thuyết (slide Ch.2), σ=1.4 tương ứng kernel 5×5 là
    chuẩn mực — đủ để triệt nhiễu nhỏ mà không làm mờ cạnh đáng kể.
    """
    return cv2.GaussianBlur(img, ksize, sigma)


def to_gray(img):
    """Chuyển BGR → Grayscale để giảm chiều dữ liệu trước khi xử lý."""
    if len(img.shape) == 2:
        return img  # đã là grayscale
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def clahe_equalization(gray_img, clip_limit=2.0, tile_size=(8, 8)):
    """
    Toán tử điểm phi tuyến (Ch.2): CLAHE — Contrast Limited Adaptive
    Histogram Equalization.
    Ưu điểm so với equalizeHist toàn cục: cân bằng cục bộ theo từng
    tile (8×8), tránh khuếch đại nhiễu ở vùng đồng đều. Phù hợp với
    cảnh đám đông có ánh sáng không đều (trong nhà, ngoài trời).
    clip_limit=2.0 giới hạn mức khuếch đại để tránh nhiễu.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray_img)


def canny_edge(gray_img, low=50, high=150):
    """
    Phát hiện cạnh (Ch.2/Ch.3): Canny edge detector.
    Pipeline Canny: Gaussian blur → Sobel gradient → NMS → Hysteresis.
    Tỉ lệ high/low = 3:1 theo khuyến nghị của Canny (1986).
    Kết quả edge_map hỗ trợ segmentation và phân tích đường viền.
    
    Trả về:
    - edge_map: ảnh nhị phân chứa các cạnh phát hiện
    - contours: danh sách các viền tìm được
    """
    edge_map = cv2.Canny(gray_img, low, high)
    contours, _ = cv2.findContours(edge_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return edge_map, contours


def normalize_frame(img):
    """
    Chuẩn hóa giá trị pixel về [0, 1] kiểu float32.
    Dùng khi cần tính toán số học (HOG, so sánh histogram).
    """
    return img.astype(np.float32) / 255.0


# ─────────────────────────────────────────────
#  Ch.3 — Phát hiện và mô tả đặc trưng
# ─────────────────────────────────────────────

def hog_descriptor(gray_img):
    """
    HOG — Histogram of Oriented Gradients (Ch.3, Dalal & Triggs 2005).

    Lý do chọn HOG cho bài toán phát hiện người:
      - Mô tả hình dạng cục bộ qua phân phối hướng gradient
      - Bất biến với thay đổi ánh sáng nhờ chuẩn hóa theo block
      - Hiệu quả với đặc trưng cấu trúc cơ thể người (đầu, vai, tay)

    Tham số chuẩn theo paper gốc:
      winSize    = (64, 128) : cửa sổ phát hiện người đứng
      blockSize  = (16, 16)  : block chuẩn hóa
      blockStride= (8, 8)    : bước trượt (overlap 50%)
      cellSize   = (8, 8)    : ô tính histogram
      nbins      = 9         : 9 bin góc từ 0°–180°

    Returns:
        descriptor (np.ndarray): vector đặc trưng 3780 chiều
        hog_image  (np.ndarray): ảnh HOG trực quan hóa (dùng debug)
    """
    win_size     = (64, 128)
    block_size   = (16, 16)
    block_stride = (8, 8)
    cell_size    = (8, 8)
    nbins        = 9

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

    # Resize về đúng kích thước cửa sổ HOG
    resized = cv2.resize(gray_img, win_size)

    descriptor = hog.compute(resized)  # shape: (3780, 1)

    # Trực quan hóa gradient để debug / báo cáo
    hog_image = _visualize_hog(resized, cell_size, nbins)

    return descriptor.flatten(), hog_image


def _visualize_hog(gray_img, cell_size=(8, 8), nbins=9):
    """
    Tạo ảnh trực quan hóa HOG (gradient orientation per cell).
    Dùng để minh họa trong báo cáo, không dùng trong pipeline chính.
    """
    gx = cv2.Sobel(gray_img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray_img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    h, w = gray_img.shape
    cx, cy = cell_size
    n_cells_x = w // cx
    n_cells_y = h // cy

    hog_img = np.zeros_like(gray_img, dtype=np.float32)

    for i in range(n_cells_y):
        for j in range(n_cells_x):
            cell_mag   = mag[i*cy:(i+1)*cy, j*cx:(j+1)*cx]
            cell_angle = angle[i*cy:(i+1)*cy, j*cx:(j+1)*cx] % 180

            hist, _ = np.histogram(cell_angle, bins=nbins,
                                   range=(0, 180), weights=cell_mag)
            dominant_bin = np.argmax(hist)
            dominant_angle = (dominant_bin / nbins) * 180

            # Vẽ đoạn thẳng thể hiện hướng gradient chính
            cx_center = j * cx + cx // 2
            cy_center = i * cy + cy // 2
            length = 3
            rad = np.deg2rad(dominant_angle)
            x1 = int(cx_center - length * np.cos(rad))
            y1 = int(cy_center - length * np.sin(rad))
            x2 = int(cx_center + length * np.cos(rad))
            y2 = int(cy_center + length * np.sin(rad))

            intensity = float(np.mean(cell_mag)) / 255.0
            cv2.line(hog_img, (x1, y1), (x2, y2), intensity * 255, 1)

    return hog_img.astype(np.uint8)


def extract_hog_from_bbox(gray_img, bbox):
    """
    Trích xuất HOG descriptor từ vùng bounding box.
    Dùng trong pipeline: sau khi YOLO phát hiện người,
    trích HOG để mô tả đặc trưng cho từng detection.

    Args:
        gray_img: ảnh grayscale toàn khung
        bbox    : tuple (x1, y1, x2, y2)

    Returns:
        descriptor (np.ndarray | None)
    """
    x1, y1, x2, y2 = bbox
    # Đảm bảo bbox nằm trong ảnh
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(gray_img.shape[1], x2), min(gray_img.shape[0], y2)

    if x2 - x1 < 16 or y2 - y1 < 16:
        return None  # vùng quá nhỏ

    crop = gray_img[y1:y2, x1:x2]
    descriptor, _ = hog_descriptor(crop)
    return descriptor


# ─────────────────────────────────────────────
#  Pipeline tổng hợp
# ─────────────────────────────────────────────

def preprocess_frame(frame):
    """
    Pipeline tiền xử lý đầy đủ cho mỗi frame video.

    Thứ tự xử lý (theo lý thuyết Ch.2):
      B1: Resize          — giảm tải tính toán, chuẩn hóa kích thước
      B2: Gaussian Blur   — lọc nhiễu trước khi tính gradient
      B3: Grayscale       — giảm chiều, tập trung vào cường độ sáng
      B4: CLAHE           — cân bằng sáng cục bộ cho cảnh không đều sáng

    Returns:
        gray_eq      (np.ndarray H×W)     : grayscale + CLAHE → MOG2, HOG
        frame_resized(np.ndarray H×W×3)  : ảnh màu → YOLO detection
    """
    frame_resized = resize_keep_ratio(frame, width=800)
    blurred       = gaussian_blur(frame_resized, ksize=(5, 5), sigma=1.4)
    gray          = to_gray(blurred)
    gray_eq       = clahe_equalization(gray, clip_limit=2.0, tile_size=(8, 8))

    return gray_eq, frame_resized


# ─────────────────────────────────────────────
#  Demo / test độc lập
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.dirname(__file__))
    from video_reader import mo_video, lay_frame

    BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = os.path.join(BASE_DIR, "data", "videos", "video1.mp4")

    cap = mo_video(video_path)
    frame_count = 0

    print("Nhấn 'q' để thoát, 'h' để xem HOG của frame hiện tại")

    while True:
        frame = lay_frame(cap)
        if frame is None:
            break

        gray_eq, edge_map, color = preprocess_frame(frame)

        cv2.imshow("1. Original (resized)", color)
        cv2.imshow("2. Gray + CLAHE (Ch.2)", gray_eq)
        cv2.imshow("3. Canny Edges (Ch.2/3)", edge_map)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            # Demo HOG trên frame hiện tại
            descriptor, hog_vis = hog_descriptor(gray_eq)
            print(f"HOG descriptor shape: {descriptor.shape}")  # (3780,)
            cv2.imshow("4. HOG Visualization (Ch.3)", hog_vis)

        frame_count += 1

    print(f"Đã xử lý {frame_count} frames.")
    cap.release()
    cv2.destroyAllWindows()