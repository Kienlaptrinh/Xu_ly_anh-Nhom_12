import cv2
import os
import sys


def mo_video(duong_dan_video):
    if not os.path.exists(duong_dan_video):
        raise FileNotFoundError(f"Không tìm thấy file: '{duong_dan_video}'")
    cap = cv2.VideoCapture(duong_dan_video)
    if not cap.isOpened():
        raise ValueError(f"Không thể mở video: '{duong_dan_video}'")
    return cap


def lay_frame(cap):
    ret, frame = cap.read()
    return frame if ret else None


def lay_thong_tin_video(cap):
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return {
        "fps"          : fps,
        "total_frames" : total,
        "width"        : width,
        "height"       : height,
        "duration_s"   : round(total / fps, 2) if fps > 0 else 0
    }


def tua_ve_dau(cap):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


if __name__ == "__main__":
    sys.path.append(os.path.dirname(__file__))
    from preprocessing import resize_keep_ratio

    BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = os.path.join(BASE_DIR, "data", "videos", "video1.mp4")

    cap  = mo_video(video_path)
    info = lay_thong_tin_video(cap)
    print(f"[INFO] Video info: {info}")
    delay = int(1000 / info["fps"]) if info["fps"] > 0 else 30

    while True:
        frame = lay_frame(cap)
        if frame is None:
            break
        cv2.imshow("Preview", resize_keep_ratio(frame))
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()