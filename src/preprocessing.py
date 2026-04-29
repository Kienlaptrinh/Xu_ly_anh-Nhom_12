import cv2
import sys
import os

# ========================
# GAUSSIAN BLUR
# ========================
def gaussian_blur(img, ksize=(5, 5)):
    return cv2.GaussianBlur(img, ksize, 0)


# ========================
# HISTOGRAM EQUALIZATION (toàn cục)
# ========================
def histogram_equalization_gray(gray_img):
    return cv2.equalizeHist(gray_img)


# ========================
# CLAHE - Histogram Equalization cục bộ (khuyến nghị)
# ========================
def clahe_equalization(gray_img):
    """Cân bằng histogram cục bộ — hiệu quả hơn cho video đám đông"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_img)


# ========================
# CHUYỂN GRAY
# ========================
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ========================
# RESIZE GIỮ TỈ LỆ
# ========================
def resize_keep_ratio(img, width=800):
    h, w = img.shape[:2]
    scale = width / w
    new_h = int(h * scale)
    return cv2.resize(img, (width, new_h))


# ========================
# PIPELINE PREPROCESS
# ========================
def preprocess_frame(frame):
    """
    Trả về:
    - gray_eq: ảnh grayscale đã xử lý (dùng cho HOG, segmentation)
    - frame_resized: ảnh màu (dùng cho YOLO)
    """
    frame_resized = resize_keep_ratio(frame)
    blurred = gaussian_blur(frame_resized)
    gray = to_gray(blurred)
    gray_eq = clahe_equalization(gray)  # CLAHE thay vì equalizeHist
    return gray_eq, frame_resized


# ========================
# TEST
# ========================
if __name__ == "__main__":
    sys.path.append(os.path.dirname(__file__))
    from video_reader import mo_video, lay_frame

    # Đường dẫn tuyệt đối tính từ vị trí file này (src/)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = os.path.join(BASE_DIR, "data", "videos", "video2.mp4")

    cap = mo_video(video_path)

    while True:
        frame = lay_frame(cap)
        if frame is None:
            break

        gray_eq, color = preprocess_frame(frame)

        cv2.imshow("Original", resize_keep_ratio(frame))
        cv2.imshow("Gray + CLAHE", gray_eq)
        cv2.imshow("Color (for YOLO)", color)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()