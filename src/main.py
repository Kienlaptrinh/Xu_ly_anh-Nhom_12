import cv2
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from ultralytics import YOLO
from tracker      import CentroidTracker
from preprocessing import preprocess_frame
from segmentation  import tao_background_subtractor, segmentation_pipeline
from video_reader  import mo_video, lay_frame, lay_thong_tin_video

# ========================
# CẤU HÌNH
# ========================
VIDEO_PATH  = os.path.join(BASE_DIR, "data", "videos", "video1.mp4")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "yolov8n.pt")
OUTPUT_PATH = os.path.join(BASE_DIR, "output", "result.avi")
LINE_Y      = 260
CONF        = 0.4
REAL_COUNT  = 50    # Ground truth — thay bằng giá trị thực của từng video

COLOR_MAP = {
    "YOLO": (0, 255, 0),
    "HOG" : (0, 165, 255),
    "MOG2": (255, 0, 0),
    "None": (128, 128, 128)
}

# ========================
# KHỞI TẠO
# ========================
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

print("[INFO] Loading YOLO model...")
model = YOLO(MODEL_PATH)

print("[INFO] Initializing HOG detector...")
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

print("[INFO] Initializing MOG2 background subtractor...")
subtractor = tao_background_subtractor()

print("[INFO] Initializing Centroid Tracker...")
ct = CentroidTracker(maxDisappeared=20, maxDistance=80, line_y=LINE_Y)

print("[INFO] Opening video...")
cap  = mo_video(VIDEO_PATH)
info = lay_thong_tin_video(cap)
print(f"[INFO] Video info: {info}")

delay  = int(1000 / info["fps"]) if info["fps"] > 0 else 30
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out    = None

# ========================
# MAIN LOOP
# ========================
while cap.isOpened():
    frame = lay_frame(cap)
    if frame is None:
        break

    # Bước 1: Preprocessing (Ch.2)
    gray_eq, edge_map, frame_resized = preprocess_frame(frame)

    if out is None:
        h, w = frame_resized.shape[:2]
        out  = cv2.VideoWriter(OUTPUT_PATH, fourcc, info["fps"], (w, h))
        if not out.isOpened():
            print("[ERROR] Không tạo được file output!")

    h, w = frame_resized.shape[:2]

    # Bước 2: Segmentation — MOG2 + Morphology + Canny edge (Ch.3 + Ch.4)
    fg_mask, rects_mog2 = segmentation_pipeline(subtractor, frame_resized, edge_map)

    # Bước 3: HOG + SVM detection (Ch.3 + Ch.5)
    boxes_hog, _ = hog.detectMultiScale(
        gray_eq, winStride=(8, 8), padding=(4, 4), scale=1.05
    )
    rects_hog = [(x, y, x+bw, y+bh) for (x, y, bw, bh) in boxes_hog]

    # Bước 4: YOLO detection (Ch.5)
    results    = model.predict(frame_resized, classes=[0], conf=CONF, verbose=False)
    rects_yolo = [tuple(box.astype("int")) for r in results for box in r.boxes.xyxy.cpu().numpy()]

    # Bước 5: Fusion — fallback chain YOLO → HOG → MOG2
    if len(rects_yolo) > 0:
        rects_final, detect_mode = rects_yolo, "YOLO"
    elif len(rects_hog) > 0:
        rects_final, detect_mode = rects_hog,  "HOG"
    elif len(rects_mog2) > 0:
        rects_final, detect_mode = rects_mog2, "MOG2"
    else:
        rects_final, detect_mode = [],          "None"

    # Bước 6: Tracking — Centroid Tracker (Ch.4)
    objects = ct.update(rects_final)

    # ========================
    # VISUALIZATION
    # ========================
    color = COLOR_MAP[detect_mode]

    for (x1, y1, x2, y2) in rects_final:
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)

    for (objectID, centroid) in objects.items():
        cv2.circle(frame_resized, tuple(centroid), 4, (0, 255, 0), -1)
        cv2.putText(frame_resized, f"ID:{objectID}",
                    (centroid[0]-10, centroid[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Counting line
    cv2.line(frame_resized, (0, LINE_Y), (w, LINE_Y), (0, 0, 255), 2)

    # HUD
    cv2.putText(frame_resized, f"Count: {ct.totalCounted}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    overlay = frame_resized.copy()
    cv2.rectangle(overlay, (15, 60), (280, 145), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame_resized, 0.55, 0, frame_resized)

    cv2.putText(frame_resized, f"Mode : {detect_mode}",
                (25, 82),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(frame_resized, f"MAE  : {abs(REAL_COUNT - ct.totalCounted)}",
                (25, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    for i, (label, c) in enumerate([("YOLO",(0,255,0)), ("HOG",(0,165,255)), ("MOG2",(255,0,0))]):
        cv2.putText(frame_resized, label, (w-80, 25+i*20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

    out.write(frame_resized)
    cv2.imshow("People Tracking & Counting", frame_resized)
    cv2.imshow("MOG2 Mask",   fg_mask)
    cv2.imshow("Canny Edges", edge_map)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# ========================
# CLEANUP
# ========================
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()

# ========================
# EVALUATION
# ========================
mae      = abs(REAL_COUNT - ct.totalCounted)
accuracy = (1 - mae / REAL_COUNT) * 100 if REAL_COUNT > 0 else 0

tp = min(ct.totalCounted, REAL_COUNT)
fp = max(0, ct.totalCounted - REAL_COUNT)
fn = max(0, REAL_COUNT - ct.totalCounted)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("\n--- KẾT QUẢ ĐÁNH GIÁ ---")
print(f"Số người thực tế  : {REAL_COUNT}")
print(f"Số người AI đếm   : {ct.totalCounted}")
print(f"MAE               : {mae}")
print(f"Độ chính xác      : {accuracy:.2f}%")
print(f"Precision         : {precision:.4f}")
print(f"Recall            : {recall:.4f}")
print(f"F1-score          : {f1:.4f}")
print(f"Video kết quả     : {OUTPUT_PATH}")