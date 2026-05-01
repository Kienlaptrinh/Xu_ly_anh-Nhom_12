import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(__file__))
from preprocessing import preprocess_frame
from video_reader import mo_video, lay_frame, lay_thong_tin_video


def tao_background_subtractor(history=500, var_threshold=50):
    """
    MOG2: học background từ `history` frame.
    detectShadows=True → shadow được đánh dấu pixel=127, loại bỏ ở bước sau.
    """
    return cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=True
    )


def ap_dung_mog2(subtractor, frame):
    fg_mask = subtractor.apply(frame)
    # Chỉ giữ foreground thực (255), loại shadow (127)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    return fg_mask


def morphology_xu_ly(fg_mask):
    """
    erode  → loại nhiễu nhỏ
    dilate → nối vùng bị đứt
    close  → lấp lỗ hổng bên trong vùng người
    """
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.erode(fg_mask,  kernel, iterations=1)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return fg_mask


def ket_hop_edge_mask(fg_mask, edge_map):
    """
    Kết hợp MOG2 mask với Canny edge (Ch.3 + Ch.4):
    Giữ vùng foreground chỉ khi có cạnh rõ → giảm false positive.
    """
    kernel       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edge_dilated = cv2.dilate(edge_map, kernel, iterations=2)
    combined     = cv2.bitwise_and(fg_mask, edge_dilated)
    combined     = cv2.dilate(combined, kernel, iterations=3)
    return combined


def lay_contours_nguoi(fg_mask, min_area=800, aspect_ratio_min=1.2):
    """
    Lọc contour theo diện tích tối thiểu và tỉ lệ h/w
    (người đứng thẳng thường có h > w).
    """
    contours, _ = cv2.findContours(
        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    rects = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if h / (w + 1e-5) < aspect_ratio_min:
            continue
        rects.append((x, y, x + w, y + h))
    return rects


def segmentation_pipeline(subtractor, frame, edge_map=None):
    """
    Pipeline phân đoạn (Ch.4):
      1. MOG2 → foreground mask (loại shadow)
      2. Morphology → làm sạch mask
      3. Kết hợp Canny edge nếu có (Ch.3)
      4. Lọc contour → bounding boxes

    Returns:
        fg_mask : mask sau xử lý
        rects   : list (x1, y1, x2, y2)
    """
    fg_mask = ap_dung_mog2(subtractor, frame)
    fg_mask = morphology_xu_ly(fg_mask)
    if edge_map is not None:
        fg_mask = ket_hop_edge_mask(fg_mask, edge_map)
    rects = lay_contours_nguoi(fg_mask)
    return fg_mask, rects


if __name__ == "__main__":
    BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = os.path.join(BASE_DIR, "data", "videos", "video1.mp4")

    cap        = mo_video(video_path)
    info       = lay_thong_tin_video(cap)
    subtractor = tao_background_subtractor()
    delay      = int(1000 / info["fps"]) if info["fps"] > 0 else 30

    while True:
        frame = lay_frame(cap)
        if frame is None:
            break
        gray_eq, edge_map, frame_resized = preprocess_frame(frame)
        fg_mask, rects = segmentation_pipeline(subtractor, frame_resized, edge_map)
        for (x1, y1, x2, y2) in rects:
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imshow("MOG2 boxes",       frame_resized)
        cv2.imshow("MOG2 Mask",        fg_mask)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()