import cv2
import numpy as np


def gaussian_blur(img, ksize=(5, 5), sigma=0):
    return cv2.GaussianBlur(img, ksize, sigma)


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def clahe_equalization(gray_img, clip_limit=2.0, tile_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray_img)


def canny_edge(gray_img, low=50, high=150):
    return cv2.Canny(gray_img, low, high)


def resize_keep_ratio(img, width=800):
    h, w = img.shape[:2]
    if w == width:
        return img
    scale = width / w
    return cv2.resize(img, (width, int(h * scale)), interpolation=cv2.INTER_LINEAR)


def normalize_frame(img):
    return img.astype(np.float32) / 255.0


def preprocess_frame(frame):
    """
    Pipeline tiền xử lý (Ch.2):
      B1: Resize giữ tỉ lệ
      B2: Gaussian Blur — giảm nhiễu
      B3: Grayscale
      B4: CLAHE — cân bằng sáng cục bộ (hiệu quả hơn equalizeHist cho cảnh đám đông)
      B5: Canny — phát hiện cạnh (Ch.3), hỗ trợ segmentation

    Returns:
        gray_eq      : grayscale + CLAHE  → HOG, MOG2
        edge_map     : Canny edges        → segmentation
        frame_resized: ảnh màu            → YOLO
    """
    frame_resized = resize_keep_ratio(frame, width=800)
    blurred       = gaussian_blur(frame_resized, ksize=(5, 5))
    gray          = to_gray(blurred)
    gray_eq       = clahe_equalization(gray)
    edge_map      = canny_edge(gray_eq, low=50, high=150)
    return gray_eq, edge_map, frame_resized


if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(__file__))
    from video_reader import mo_video, lay_frame

    BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = os.path.join(BASE_DIR, "data", "videos", "video3.mp4")

    cap = mo_video(video_path)
    while True:
        frame = lay_frame(cap)
        if frame is None:
            break
        gray_eq, edge_map, color = preprocess_frame(frame)
        cv2.imshow("Original (resized)", color)
        cv2.imshow("Gray + CLAHE",       gray_eq)
        cv2.imshow("Canny Edges (Ch.3)", edge_map)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()