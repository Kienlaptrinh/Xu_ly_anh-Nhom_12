import cv2
import os
import sys

sys.path.append(os.path.dirname(__file__))
from preprocessing import preprocess_frame
from video_reader import mo_video, lay_frame, lay_thong_tin_video


def tao_background_subtractor():
    return cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=True
    )


def ap_dung_mog2(subtractor, frame):
    fg_mask = subtractor.apply(frame)
    return fg_mask


def morphology_xu_ly(fg_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
    return fg_mask


def lay_contours_nguoi(fg_mask, min_area=500):
    contours, _ = cv2.findContours(
        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    rects = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            rects.append((x, y, x + w, y + h))
    return rects


def segmentation_pipeline(subtractor, frame):
    fg_mask = ap_dung_mog2(subtractor, frame)
    fg_mask = morphology_xu_ly(fg_mask)
    rects = lay_contours_nguoi(fg_mask)
    return fg_mask, rects


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = os.path.join(BASE_DIR, "data", "videos", "video1.mp4")

    cap = mo_video(video_path)
    info = lay_thong_tin_video(cap)
    print(f"[INFO] Video info: {info}")

    subtractor = tao_background_subtractor()
    delay = int(1000 / info["fps"]) if info["fps"] > 0 else 30

    while True:
        frame = lay_frame(cap)
        if frame is None:
            break

        _, frame_resized = preprocess_frame(frame)
        fg_mask, rects = segmentation_pipeline(subtractor, frame_resized)

        for (x1, y1, x2, y2) in rects:
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow("Frame", frame_resized)
        cv2.imshow("MOG2 Mask", fg_mask)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()