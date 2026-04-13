import cv2

# ========================
# GAUSSIAN BLUR
# ========================
def gaussian_blur(img, ksize=(5, 5)):
    return cv2.GaussianBlur(img, ksize, 0)


# ========================
# HISTOGRAM EQUALIZATION
# ========================
def histogram_equalization_gray(gray_img):
    return cv2.equalizeHist(gray_img)


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
# PIPELINE PREPROCESS (CHUẨN NHẤT)
# ========================
def preprocess_frame(frame):
    """
    Trả về:
    - gray_eq: ảnh grayscale đã xử lý (dùng cho HOG, segmentation)
    - frame_resized: ảnh màu (dùng cho YOLO)
    """

    # Resize trước
    frame_resized = resize_keep_ratio(frame)

    # Blur
    blurred = gaussian_blur(frame_resized)

    # Gray
    gray = to_gray(blurred)

    # Histogram Equalization
    gray_eq = histogram_equalization_gray(gray)

    return gray_eq, frame_resized


# ========================
# TEST
# ========================
if __name__ == "__main__":
    from doc_video import mo_video, lay_frame

    cap = mo_video("du_lieu/video_test.mp4")

    while True:
        frame = lay_frame(cap)
        if frame is None:
            break

        gray_eq, color = preprocess_frame(frame)

        cv2.imshow("Original", resize_keep_ratio(frame))
        cv2.imshow("Gray Equalized", gray_eq)
        cv2.imshow("Color (for YOLO)", color)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()