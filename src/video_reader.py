import cv2
import os
import sys

def mo_video(duong_dan_video):
    cap = cv2.VideoCapture(duong_dan_video)
    if not cap.isOpened():
        raise ValueError(f"Lỗi: Không thể mở video tại '{duong_dan_video}'!")
    return cap

def lay_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def lay_thong_tin_video(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return {"fps": fps, "total_frames": total_frames,
            "width": width, "height": height}

if __name__ == "__main__":
    sys.path.append(os.path.dirname(__file__))
    from preprocessing import resize_keep_ratio

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = os.path.join(BASE_DIR, "data", "videos", "video1.mp4")

    cap_test = mo_video(video_path)
    info = lay_thong_tin_video(cap_test)
    print(f"Thông tin video: {info}")
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