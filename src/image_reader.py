"""
image_reader.py — Đọc và xử lý ảnh tĩnh
==========================================
Chương 2, 3, 4, 5 — Pipeline đầy đủ trên ảnh tĩnh:
  - Preprocessing (Ch.2)    : resize, blur, CLAHE, Canny
  - HOG Detection (Ch.3)    : phát hiện người bằng HOG + SVM detector
  - MOG2 Segmentation (Ch.4): background subtraction (fallback)
  - YOLO Detection (Ch.5)   : phát hiện người bằng YOLOv8 với tiling
  - Fallback chain          : YOLO → HOG → MOG2

Tiling strategy cho ảnh độ phân giải cao:
  Ảnh lớn (> 640px) được chia thành các tile 640×640 có overlap 200px.
  Overlap giúp phát hiện người nằm ở ranh giới tile.
  Sau khi gộp kết quả → áp NMS để loại bounding box trùng.
"""

import cv2
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

# ── Tham số HOG ──
HOG_WIN_STRIDE = (4, 4)    # stride nhỏ → phát hiện nhiều vị trí hơn, chậm hơn
HOG_PADDING    = (8, 8)    # padding quanh cửa sổ → phát hiện người ở rìa ảnh
HOG_SCALE      = 1.03      # scale pyramid step nhỏ → không bỏ sót người nhỏ/lớn

# ── Tham số YOLO tiling ──
TILE_SIZE      = 640       # kích thước tile khớp với input size của YOLOv8
TILE_OVERLAP   = 200       # overlap để không bỏ sót người ở ranh giới tile
NMS_THRESHOLD  = 0.4       # IoU threshold cho NMS (loại box trùng > 40%)


# ─────────────────────────────────────────────
#  Đọc ảnh
# ─────────────────────────────────────────────

def doc_anh(duong_dan_anh):
    """
    Đọc ảnh từ file, raise lỗi rõ ràng nếu không tìm thấy.

    Returns:
        np.ndarray (BGR)
    """
    if not os.path.exists(duong_dan_anh):
        raise FileNotFoundError(f"Không tìm thấy file: '{duong_dan_anh}'")
    anh = cv2.imread(duong_dan_anh)
    if anh is None:
        raise ValueError(f"Không đọc được ảnh (format không hỗ trợ?): '{duong_dan_anh}'")
    return anh


def lay_thong_tin_anh(anh):
    """Trả về dict thông tin cơ bản của ảnh."""
    h, w = anh.shape[:2]
    return {
        "width"    : w,
        "height"   : h,
        "channels" : anh.shape[2] if len(anh.shape) == 3 else 1,
        "size_px"  : w * h,
        "dtype"    : str(anh.dtype)
    }


# ─────────────────────────────────────────────
#  NMS — Non-Maximum Suppression
# ─────────────────────────────────────────────

def nms_thu_cong(rects, nms_threshold=NMS_THRESHOLD):
    """
    Áp dụng NMS để loại bounding box trùng nhau.

    Dùng cv2.dnn.NMSBoxes thay vì implement tay:
    - Nhanh hơn, được tối ưu trong OpenCV
    - Nhận format (x, y, w, h) nên cần convert từ (x1,y1,x2,y2)

    Args:
        rects         : list of (x1, y1, x2, y2)
        nms_threshold : IoU threshold — box có IoU > threshold bị loại

    Returns:
        list of (x1, y1, x2, y2) sau NMS
    """
    if len(rects) == 0:
        return []

    # Convert sang (x, y, w, h) cho NMSBoxes
    boxes  = np.array(
        [[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2) in rects],
        dtype=np.float32
    )
    # Confidence = 1.0 cho tất cả (không có score từ HOG/MOG2)
    scores  = np.ones(len(boxes), dtype=np.float32)
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(),
        score_threshold=0.0,
        nms_threshold=nms_threshold
    )

    if len(indices) == 0:
        return []
    return [rects[i] for i in indices.flatten()]


# ─────────────────────────────────────────────
#  Ch.3 — HOG Detection
# ─────────────────────────────────────────────

def phat_hien_hog(hog, gray_eq):
    """
    Phát hiện người bằng HOG + SVM detector (Ch.3).

    HOG (Histogram of Oriented Gradients) + Linear SVM:
      - Trượt cửa sổ 64×128 trên ảnh ở nhiều tỉ lệ (scale pyramid)
      - Mỗi cửa sổ: tính HOG descriptor → SVM phân loại người/không-người
      - winStride=(4,4): bước trượt nhỏ → phát hiện chính xác hơn
      - scale=1.03: pyramid dày → không bỏ sót kích thước người

    Lý do dùng HOG trước YOLO trong fallback:
      - HOG không cần GPU, chạy được trên CPU yếu
      - Phù hợp ảnh tĩnh độ phân giải trung bình
      - Cung cấp đại diện rõ ràng cho Ch.3 trong pipeline

    Returns:
        rects_hog: list of (x1, y1, x2, y2) sau NMS
    """
    boxes, _ = hog.detectMultiScale(
        gray_eq,
        winStride=HOG_WIN_STRIDE,
        padding=HOG_PADDING,
        scale=HOG_SCALE
    )

    if len(boxes) == 0:
        return []

    # Convert từ (x, y, w, h) sang (x1, y1, x2, y2)
    rects_raw = [(x, y, x + bw, y + bh) for (x, y, bw, bh) in boxes]
    return nms_thu_cong(rects_raw)


# ─────────────────────────────────────────────
#  Ch.5 — YOLO Detection với Tiling
# ─────────────────────────────────────────────

def detect_yolo_tiling(model, frame, conf=0.15):
    """
    Phát hiện người bằng YOLOv8 với chiến lược tiling (Ch.5).

    Vấn đề: YOLOv8 mặc định resize ảnh về 640×640.
    Với ảnh lớn, người nhỏ ở xa bị shrink → bỏ sót.

    Giải pháp — Tiling:
      1. Chia ảnh thành các tile 640×640 có overlap TILE_OVERLAP px
      2. Chạy YOLO trên từng tile
      3. Cộng offset (x, y) vào tọa độ box để về tọa độ ảnh gốc
      4. Gộp tất cả box → áp NMS để loại trùng

    Overlap cần thiết vì người có thể nằm ở ranh giới tile.
    conf=0.15 thấp hơn mặc định (0.25) để không bỏ sót người nhỏ,
    NMS sau đó lọc lại false positive.

    Returns:
        rects_all: list of (x1, y1, x2, y2) sau NMS
    """
    h, w      = frame.shape[:2]
    rects_all = []

    for y in range(0, h, TILE_SIZE - TILE_OVERLAP):
        for x in range(0, w, TILE_SIZE - TILE_OVERLAP):
            tile    = frame[y:min(y + TILE_SIZE, h), x:min(x + TILE_SIZE, w)]
            results = model.predict(
                tile, classes=[0], conf=conf, iou=0.4, verbose=False
            )
            for r in results:
                for box in r.boxes.xyxy.cpu().numpy():
                    bx1, by1, bx2, by2 = box.astype("int")
                    # Cộng offset tile về tọa độ ảnh gốc
                    rects_all.append((bx1 + x, by1 + y, bx2 + x, by2 + y))

    return nms_thu_cong(rects_all)


# ─────────────────────────────────────────────
#  Pipeline tổng hợp
# ─────────────────────────────────────────────

def xu_ly_anh_tinh(anh, model, hog, subtractor, conf=0.25):
    """
    Pipeline phát hiện người đầy đủ trên ảnh tĩnh.

    Tích hợp Ch.2 + Ch.3 + Ch.4 + Ch.5:
      Ch.2: preprocess_frame (resize, blur, CLAHE, Canny)
      Ch.3: HOG + SVM detector
      Ch.4: MOG2 segmentation (lưu ý: hiệu quả thấp trên ảnh tĩnh
            vì không có lịch sử frame — dùng như fallback minh họa)
      Ch.5: YOLOv8 với tiling

    Fallback chain: YOLO → HOG → MOG2
      Ưu tiên YOLO vì accuracy cao nhất.
      HOG là backup khi YOLO không detect được gì (người quá nhỏ/mờ).
      MOG2 là last resort + minh họa kỹ thuật Ch.4.

    Args:
        anh        : ảnh BGR gốc
        model      : YOLOv8 model object
        hog        : cv2.HOGDescriptor với SVM detector
        subtractor : MOG2 background subtractor
        conf       : YOLO confidence threshold

    Returns:
        output      : ảnh đã vẽ bounding boxes
        rects_final : list (x1,y1,x2,y2) kết quả cuối
        detect_mode : 'YOLO' | 'HOG' | 'MOG2' | 'None'
        edge_map    : Canny edge map (dùng visualize/báo cáo)
        fg_mask     : MOG2 foreground mask (dùng visualize/báo cáo)
        all_results : dict chứa kết quả từng phương pháp (dùng so sánh)
    """
    from preprocessing import preprocess_frame
    from segmentation  import segmentation_pipeline

    # Ch.2 — Tiền xử lý
    gray_eq, edge_map, frame_resized = preprocess_frame(anh)

    # Ch.4 — MOG2 segmentation
    fg_mask, rects_mog2 = segmentation_pipeline(
        subtractor, frame_resized, edge_map
    )

    # Ch.3 — HOG detection
    rects_hog = phat_hien_hog(hog, gray_eq)

    # Ch.5 — YOLO detection với tiling
    rects_yolo = detect_yolo_tiling(model, frame_resized, conf=conf)

    # ── Fallback chain ──
    if len(rects_yolo) > 0:
        rects_final, detect_mode = rects_yolo, "YOLO"
    elif len(rects_hog) > 0:
        rects_final, detect_mode = rects_hog,  "HOG"
    elif len(rects_mog2) > 0:
        rects_final, detect_mode = rects_mog2, "MOG2"
    else:
        rects_final, detect_mode = [],          "None"

    # ── Visualize ──
    color_map = {
        "YOLO": (0, 255, 0),    # xanh lá
        "HOG" : (0, 165, 255),  # cam
        "MOG2": (255, 0, 0),    # xanh lam
        "None": (128, 128, 128) # xám
    }
    output = frame_resized.copy()
    color  = color_map[detect_mode]

    for (x1, y1, x2, y2) in rects_final:
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

    label = f"Detected: {len(rects_final)} | Mode: {detect_mode}"
    cv2.putText(output, label, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Dict kết quả từng phương pháp — dùng cho evaluation
    all_results = {
        "YOLO": rects_yolo,
        "HOG" : rects_hog,
        "MOG2": rects_mog2,
    }

    return output, rects_final, detect_mode, edge_map, fg_mask, all_results


def ve_ket_qua_tat_ca(frame_resized, all_results):
    """
    Vẽ kết quả của cả 3 phương pháp lên 3 ảnh riêng để so sánh.
    Dùng cho báo cáo — minh họa sự khác nhau giữa YOLO, HOG, MOG2.

    Returns:
        dict {'YOLO': img, 'HOG': img, 'MOG2': img}
    """
    color_map = {
        "YOLO": (0, 255, 0),
        "HOG" : (0, 165, 255),
        "MOG2": (255, 0, 0),
    }
    panels = {}
    for method, rects in all_results.items():
        panel = frame_resized.copy()
        for (x1, y1, x2, y2) in rects:
            cv2.rectangle(panel, (x1, y1), (x2, y2), color_map[method], 2)
        cv2.putText(panel, f"{method}: {len(rects)} người",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_map[method], 2)
        panels[method] = panel
    return panels


# ─────────────────────────────────────────────
#  Demo / test độc lập
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from ultralytics import YOLO
    from segmentation import tao_background_subtractor

    image_path = os.path.join(BASE_DIR, "data", "images", "pic1.jpg")
    model_path = os.path.join(BASE_DIR, "models", "yolov8n.pt")

    # Đọc ảnh
    anh = doc_anh(image_path)
    info = lay_thong_tin_anh(anh)
    print(f"[INFO] Ảnh: {info}")

    # Khởi tạo models
    model      = YOLO(model_path)
    hog        = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    subtractor = tao_background_subtractor()

    # Chạy pipeline
    output, rects, mode, edge_map, fg_mask, all_results = xu_ly_anh_tinh(
        anh, model, hog, subtractor
    )
    print(f"[RESULT] {len(rects)} người | Mode: {mode}")
    print(f"  YOLO: {len(all_results['YOLO'])} | "
          f"HOG: {len(all_results['HOG'])} | "
          f"MOG2: {len(all_results['MOG2'])}")

    # Hiển thị kết quả chính
    cv2.imshow("Detection Result (final)", output)
    cv2.imshow("Canny Edges — Ch.2/3",     edge_map)
    cv2.imshow("MOG2 Mask — Ch.4",         fg_mask)

    # So sánh từng phương pháp
    panels = ve_ket_qua_tat_ca(
        output.copy().shape and __import__('numpy').zeros_like(output),
        all_results
    )
    # Ghép 3 panel cạnh nhau
    panels_list = [all_results["YOLO"], all_results["HOG"], all_results["MOG2"]]
    compare_imgs = []
    for method, rects_m in all_results.items():
        p = anh.copy()
        color = {"YOLO":(0,255,0),"HOG":(0,165,255),"MOG2":(255,0,0)}[method]
        for (x1,y1,x2,y2) in rects_m:
            cv2.rectangle(p,(x1,y1),(x2,y2),color,2)
        cv2.putText(p,f"{method}:{len(rects_m)}",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
        from preprocessing import resize_keep_ratio
        compare_imgs.append(resize_keep_ratio(p, width=400))

    # Đảm bảo cùng height trước khi hstack
    min_h = min(img.shape[0] for img in compare_imgs)
    compare_imgs = [img[:min_h] for img in compare_imgs]
    cv2.imshow("So sánh: YOLO | HOG | MOG2",
               __import__('numpy').hstack(compare_imgs))

    cv2.waitKey(0)
    cv2.destroyAllWindows()