"""
segmentation.py — Phân đoạn ảnh (Ch.4)
=======================================
Chương 4 — Phân đoạn ảnh:
  - MOG2 Background Subtraction : phân đoạn vùng chuyển động (foreground)
  - Morphological Operations    : erosion, dilation, closing — làm sạch mask
  - Contour Analysis            : phát hiện và lọc đường viền người
  - Edge-guided Segmentation    : kết hợp Canny edge (Ch.3) để tinh chỉnh mask

Lý do chọn MOG2 thay vì phương pháp phân đoạn tĩnh (K-Means, GrabCut):
  - Video theo dõi người → background tương đối ổn định → MOG2 hiệu quả
  - MOG2 thích nghi được với thay đổi ánh sáng nhờ mô hình Gaussian hỗn hợp
  - Nhanh hơn GrabCut (real-time), không cần seed point như Active Contours
"""

import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(__file__))
from preprocessing import preprocess_frame, extract_hog_from_bbox
from video_reader import mo_video, lay_frame, lay_thong_tin_video


# ─────────────────────────────────────────────
#  MOG2 — Background Subtraction
# ─────────────────────────────────────────────

def tao_background_subtractor(history=500, var_threshold=50):
    """
    Khởi tạo MOG2 Background Subtractor (Ch.4 — phân đoạn vùng động).

    MOG2 mô hình hóa mỗi pixel bằng hỗn hợp Gaussian (Mixture of Gaussians).
    Pixel nào không khớp với background model → foreground.

    Tham số:
        history      : số frame để học background (500 ≈ ~16s ở 30fps)
                       Tăng → ổn định hơn nhưng chậm thích nghi
        var_threshold: ngưỡng Mahalanobis distance phân biệt FG/BG
                       Tăng → ít nhạy hơn (bỏ sót người đứng yên)
                       Giảm → nhạy hơn (nhiều nhiễu)
        detectShadows: True → shadow được đánh dấu pixel=127 (xám)
                       để loại riêng ở bước threshold, tránh nhầm với người
    """
    return cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=True
    )


def ap_dung_mog2(subtractor, frame):
    """
    Áp dụng MOG2 và loại bỏ shadow.

    fg_mask sau apply():
      255 = foreground (người, xe...)
      127 = shadow (loại bỏ)
      0   = background

    Threshold > 200 → chỉ giữ foreground thực sự (255).
    """
    fg_mask = subtractor.apply(frame)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    return fg_mask


# ─────────────────────────────────────────────
#  Ch.4 — Morphological Operations
# ─────────────────────────────────────────────

def morphology_xu_ly(fg_mask, kernel_size=5):
    """
    Làm sạch foreground mask bằng phép toán hình thái học (Ch.4).

    Thứ tự và lý do:
      1. Erosion (1 lần):
         - Co vùng FG → loại pixel nhiễu đơn lẻ và vùng nhỏ
         - Kernel ellipse phù hợp hơn square cho đối tượng tròn (người)

      2. Dilation (2 lần):
         - Phục hồi kích thước sau erosion + mở rộng thêm
         - Nối các vùng bị đứt do quần áo tối, bóng đổ

      3. Closing = Dilation rồi Erosion (2 lần):
         - Lấp các lỗ hổng bên trong vùng người (do màu áo sáng,
           vùng phản chiếu ánh sáng)
         - Closing tốt hơn dilation đơn thuần vì giữ nguyên kích thước ngoài

    Kernel ellipse (5×5): phù hợp với hình dạng cơ thể người,
    tránh artifact góc nhọn của kernel vuông.
    """
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Bước 1: Erosion — loại nhiễu nhỏ
    fg_mask = cv2.erode(fg_mask, kernel, iterations=1)

    # Bước 2: Dilation — nối vùng bị đứt, phục hồi kích thước
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

    # Bước 3: Closing — lấp lỗ hổng bên trong vùng người
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return fg_mask


def morphology_adaptive(fg_mask, density="normal"):
    """
    Morphology thích nghi theo mật độ người trong cảnh.

    Khi đông người (density='crowd'):
        - Kernel nhỏ hơn để tránh merge hai người thành một vùng
    Khi thưa người (density='normal'):
        - Kernel lớn hơn để lấp lỗ hổng tốt hơn

    Args:
        density: 'normal' | 'crowd'
    """
    if density == "crowd":
        kernel_size = 3  # nhỏ hơn, tránh merge người
    else:
        kernel_size = 5

    return morphology_xu_ly(fg_mask, kernel_size=kernel_size)


# ─────────────────────────────────────────────
#  Edge-guided Segmentation (Ch.3 + Ch.4)
# ─────────────────────────────────────────────

def ket_hop_edge_mask(fg_mask, edge_map):
    """
    Tinh chỉnh foreground mask bằng Canny edge map (kết hợp Ch.3 + Ch.4).

    Ý tưởng: Foreground thực sự phải có cạnh rõ ràng.
    Vùng FG không có cạnh → nhiễu (ánh sáng thay đổi, lá cây lay động...).

    Quy trình:
      1. Dilate edge_map → tạo vùng đệm quanh cạnh (5px)
         Cần thiết vì contour người không khớp pixel-perfect với FG mask
      2. bitwise_and → chỉ giữ FG có cạnh lân cận
      3. Dilate kết quả → phục hồi vùng bên trong đã bị AND thu hẹp

    Lưu ý: Không dùng AND trực tiếp mà phải dilate edge trước,
    vì edge là đường 1px còn FG mask là vùng đặc.
    """
    kernel       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Mở rộng edge thành vùng để overlap với FG mask
    edge_dilated = cv2.dilate(edge_map, kernel, iterations=2)

    # Giữ FG chỉ khi có cạnh lân cận
    combined     = cv2.bitwise_and(fg_mask, edge_dilated)

    # Phục hồi kích thước vùng FG sau AND
    combined     = cv2.dilate(combined, kernel, iterations=3)

    return combined


# ─────────────────────────────────────────────
#  Contour Analysis — Lọc và phát hiện người
# ─────────────────────────────────────────────

def lay_contours_nguoi(fg_mask, min_area=800, max_area=50000,
                        aspect_ratio_min=1.2, aspect_ratio_max=4.5):
    """
    Phân tích đường viền (contour) để xác định vùng chứa người (Ch.4).

    Lý do dùng contour thay vì chỉ dùng bounding box trực tiếp:
      - cv2.findContours cho phép tính diện tích thực (cv2.contourArea)
        chính xác hơn w*h (bounding box bao gồm cả vùng trống)
      - Có thể tính convexity, circularity để lọc thêm nếu cần

    Tiêu chí lọc:
      min_area (800px²): loại nhiễu nhỏ (lá cây, bóng nhỏ)
      max_area (50000px²): loại vùng quá lớn (toàn bộ background thay đổi)
      aspect_ratio h/w ∈ [1.2, 4.5]:
        - Người đứng thẳng: h/w ≈ 2–3
        - Người ngồi / xa camera: h/w ≈ 1.2–1.5
        - Loại vật ngang (xe, túi xách): h/w < 1.2
        - Loại noise dọc (cột, tường): h/w > 4.5

    Returns:
        rects     : list of (x1, y1, x2, y2) — bounding boxes
        contours  : list of contours tương ứng (dùng để vẽ/debug)
    """
    contours, _ = cv2.findContours(
        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    rects    = []
    kept_cnt = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Lọc theo diện tích
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Lọc theo tỉ lệ h/w
        ratio = h / (w + 1e-5)
        if ratio < aspect_ratio_min or ratio > aspect_ratio_max:
            continue

        rects.append((x, y, x + w, y + h))
        kept_cnt.append(cnt)

    return rects, kept_cnt


def tinh_mat_do_vung(fg_mask, bbox):
    """
    Tính mật độ foreground bên trong bounding box (fill ratio).
    fill_ratio cao → vùng đặc, khả năng là người cao hơn.
    fill_ratio thấp → vùng thưa, có thể là bóng hoặc nhiễu.

    Args:
        fg_mask: foreground mask nhị phân
        bbox   : (x1, y1, x2, y2)
    Returns:
        fill_ratio: float ∈ [0, 1]
    """
    x1, y1, x2, y2 = bbox
    roi  = fg_mask[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return float(np.count_nonzero(roi)) / roi.size


# ─────────────────────────────────────────────
#  Pipeline tổng hợp
# ─────────────────────────────────────────────

def segmentation_pipeline(subtractor, frame, edge_map=None,
                           use_edge=True, use_fill_filter=True,
                           fill_threshold=0.25):
    """
    Pipeline phân đoạn đầy đủ (Ch.4).

    Các bước:
      1. MOG2 → foreground mask (loại shadow pixel=127)
      2. Morphology → erosion + dilation + closing
      3. [tuỳ chọn] Kết hợp Canny edge để lọc nhiễu
      4. Contour analysis → lọc theo diện tích và tỉ lệ
      5. [tuỳ chọn] Lọc thêm theo fill ratio

    Args:
        subtractor      : MOG2 object đã khởi tạo
        frame           : frame màu (BGR)
        edge_map        : Canny edge map từ preprocessing (hoặc None)
        use_edge        : có dùng edge-guided segmentation không
        use_fill_filter : có lọc theo fill ratio không
        fill_threshold  : fill_ratio tối thiểu để giữ bbox

    Returns:
        fg_mask  (np.ndarray): mask sau xử lý (dùng để visualize)
        rects    (list)      : [(x1,y1,x2,y2), ...]
    """
    # Bước 1: MOG2
    fg_mask = ap_dung_mog2(subtractor, frame)

    # Bước 2: Morphology
    fg_mask = morphology_xu_ly(fg_mask)

    # Bước 3: Edge-guided (nếu có edge_map)
    if use_edge and edge_map is not None:
        fg_mask = ket_hop_edge_mask(fg_mask, edge_map)

    # Bước 4: Contour → bounding boxes
    rects, _ = lay_contours_nguoi(fg_mask)

    # Bước 5: Lọc theo fill ratio
    if use_fill_filter:
        rects = [
            r for r in rects
            if tinh_mat_do_vung(fg_mask, r) >= fill_threshold
        ]

    return fg_mask, rects


# ─────────────────────────────────────────────
#  Visualize helpers
# ─────────────────────────────────────────────

def ve_ket_qua(frame, rects, fg_mask, label_prefix="SEG"):
    """
    Vẽ bounding boxes từ segmentation lên frame.
    Màu xanh lam (255, 0, 0) phân biệt với YOLO (xanh lá).

    Returns:
        vis_frame: frame đã vẽ boxes
        overlay  : ảnh ghép frame + mask (dùng cho báo cáo)
    """
    vis_frame = frame.copy()

    for i, (x1, y1, x2, y2) in enumerate(rects):
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"{label_prefix}-{i+1}"
        cv2.putText(vis_frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Overlay mask (tô màu đỏ vùng FG lên frame gốc)
    mask_color = np.zeros_like(frame)
    mask_color[:, :, 2] = fg_mask  # kênh đỏ
    overlay = cv2.addWeighted(vis_frame, 0.8, mask_color, 0.3, 0)

    return vis_frame, overlay


# ─────────────────────────────────────────────
#  Demo / test độc lập
# ─────────────────────────────────────────────

if __name__ == "__main__":
    BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = os.path.join(BASE_DIR, "data", "videos", "video1.mp4")

    cap        = mo_video(video_path)
    info       = lay_thong_tin_video(cap)
    subtractor = tao_background_subtractor(history=500, var_threshold=50)
    delay      = int(1000 / info["fps"]) if info["fps"] > 0 else 30

    print("Nhấn 'q' để thoát | 'e' toggle edge-guided | 'f' toggle fill filter")
    use_edge = True
    use_fill = True

    while True:
        frame = lay_frame(cap)
        if frame is None:
            break

        gray_eq, edge_map, frame_resized = preprocess_frame(frame)
        fg_mask, rects = segmentation_pipeline(
            subtractor, frame_resized, edge_map,
            use_edge=use_edge, use_fill_filter=use_fill
        )

        vis_frame, overlay = ve_ket_qua(frame_resized, rects, fg_mask)

        # Hiển thị số người phát hiện
        cv2.putText(vis_frame, f"Detected: {len(rects)} | Edge:{use_edge} Fill:{use_fill}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Segmentation — Bounding Boxes", vis_frame)
        cv2.imshow("MOG2 Mask (Ch.4)", fg_mask)
        cv2.imshow("Overlay", overlay)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            use_edge = not use_edge
            print(f"Edge-guided: {'ON' if use_edge else 'OFF'}")
        elif key == ord('f'):
            use_fill = not use_fill
            print(f"Fill filter: {'ON' if use_fill else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()