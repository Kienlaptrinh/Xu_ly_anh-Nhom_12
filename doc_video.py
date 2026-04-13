import cv2

def mo_video(duong_dan_video):
    """
    Mở video
    """
    cap = cv2.VideoCapture(duong_dan_video)

    if not cap.isOpened():
        raise ValueError(f"Lỗi: Không thể mở video tại '{duong_dan_video}'!")

    return cap


# ========================
# LẤY FRAME (QUAN TRỌNG)
# ========================
def lay_frame(cap):
    ret, frame = cap.read()

    if not ret:
        return None

    return frame


# ========================
# RESIZE GIỮ TỈ LỆ
# ========================
def resize_keep_ratio(img, width=800):
    h, w = img.shape[:2]
    scale = width / w
    new_h = int(h * scale)

    return cv2.resize(img, (width, new_h))


# ========================
# TEST VIDEO
# ========================
if __name__ == "__main__":
    duong_dan = 'du_lieu/video_test.mp4'
    cap_test = mo_video(duong_dan)

    print("Mở video thành công! (Bấm 'q' để thoát)")

    while True:
        frame = lay_frame(cap_test)

        if frame is None:
            print("Đã phát hết video.")
            break

        frame_preview = resize_keep_ratio(frame)

        cv2.imshow('Preview Video', frame_preview)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_test.release()
    cv2.destroyAllWindows()