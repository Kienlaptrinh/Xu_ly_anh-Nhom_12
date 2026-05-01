import cv2
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

# Tham số HOG
HOG_WIN_STRIDE = (4, 4)
HOG_PADDING    = (8, 8)
HOG_SCALE      = 1.03

# Tham số YOLO tiling
TILE_SIZE      = 640
TILE_OVERLAP   = 200
NMS_THRESHOLD  = 0.4


def doc_anh(duong_dan_anh):
    if not os.path.exists(duong_dan_anh):
        raise FileNotFoundError(f"Không tìm thấy file: '{duong_dan_anh}'")
    anh = cv2.imread(duong_dan_anh)
    if anh is None:
        raise ValueError(f"Không đọc được ảnh: '{duong_dan_anh}'")
    return anh


def lay_thong_tin_anh(anh):
    h, w = anh.shape[:2]
    return {
        "width"   : w,
        "height"  : h,
        "channels": anh.shape[2] if len(anh.shape) == 3 else 1
    }


def nms_thu_cong(rects, nms_threshold=NMS_THRESHOLD):
    if len(rects) == 0:
        return []
    boxes  = np.array([[x1, y1, x2-x1, y2-y1] for (x1, y1, x2, y2) in rects], dtype=np.float32)
    scores = np.ones(len(boxes), dtype=np.float32)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(),
                                score_threshold=0.0, nms_threshold=nms_threshold)
    return [] if len(indices) == 0 else [rects[i] for i in indices.flatten()]


def detect_yolo_tiling(model, frame, conf=0.15):
    """
    Chia ảnh thành các tile có overlap để YOLO phát hiện người ở mọi vị trí,
    sau đó gộp kết quả và áp NMS để loại box trùng.
    """
    h, w      = frame.shape[:2]
    rects_all = []
    for y in range(0, h, TILE_SIZE - TILE_OVERLAP):
        for x in range(0, w, TILE_SIZE - TILE_OVERLAP):
            tile    = frame[y:min(y+TILE_SIZE, h), x:min(x+TILE_SIZE, w)]
            results = model.predict(tile, classes=[0], conf=conf, iou=0.4, verbose=False)
            for r in results:
                for box in r.boxes.xyxy.cpu().numpy():
                    bx1, by1, bx2, by2 = box.astype("int")
                    rects_all.append((bx1+x, by1+y, bx2+x, by2+y))
    return nms_thu_cong(rects_all)


def xu_ly_anh_tinh(anh, model, hog, subtractor, conf=0.25):
    """
    Pipeline phát hiện người trên ảnh tĩnh.
    Lưu ý: MOG2 trên ảnh tĩnh không có lịch sử frame nên hiệu quả thấp —
    chủ yếu dùng như fallback cuối cùng và minh họa kỹ thuật Ch.4.
    """
    from preprocessing import preprocess_frame
    from segmentation  import segmentation_pipeline

    gray_eq, edge_map, frame_resized = preprocess_frame(anh)
    fg_mask, rects_mog2 = segmentation_pipeline(subtractor, frame_resized, edge_map)

    boxes_hog, _ = hog.detectMultiScale(
        gray_eq, winStride=HOG_WIN_STRIDE, padding=HOG_PADDING, scale=HOG_SCALE
    )
    rects_hog = nms_thu_cong([(x, y, x+bw, y+bh) for (x, y, bw, bh) in boxes_hog])
    rects_yolo = detect_yolo_tiling(model, frame_resized, conf=conf)

    # Fallback chain: YOLO → HOG → MOG2
    if len(rects_yolo) > 0:
        rects_final, detect_mode = rects_yolo, "YOLO"
    elif len(rects_hog) > 0:
        rects_final, detect_mode = rects_hog,  "HOG"
    elif len(rects_mog2) > 0:
        rects_final, detect_mode = rects_mog2, "MOG2"
    else:
        rects_final, detect_mode = [], "None"

    color_map = {
        "YOLO": (0, 255, 0),
        "HOG" : (0, 165, 255),
        "MOG2": (255, 0, 0),
        "None": (128, 128, 128)
    }
    output = frame_resized.copy()
    color  = color_map[detect_mode]
    for (x1, y1, x2, y2) in rects_final:
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
    cv2.putText(output, f"Detected: {len(rects_final)} | Mode: {detect_mode}",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return output, rects_final, detect_mode, edge_map, fg_mask


if __name__ == "__main__":
    from ultralytics import YOLO
    from segmentation import tao_background_subtractor

    image_path = os.path.join(BASE_DIR, "data", "images", "pic4.jpg")
    model_path = os.path.join(BASE_DIR, "models", "yolov8n.pt")

    anh  = doc_anh(image_path)
    print(f"[INFO] Ảnh gốc: {lay_thong_tin_anh(anh)}")

    model      = YOLO(model_path)
    hog        = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    subtractor = tao_background_subtractor()

    output, rects, mode, edge_map, fg_mask = xu_ly_anh_tinh(anh, model, hog, subtractor)
    print(f"[RESULT] Detected: {len(rects)} người | Mode: {mode}")

    cv2.imshow("Detection Result",   output)
    cv2.imshow("Canny Edges (Ch.3)", edge_map)
    cv2.imshow("MOG2 Mask (Ch.4)",   fg_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()