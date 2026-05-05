import cv2
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from ultralytics import YOLO
from tracker import CentroidTracker
from preprocessing import preprocess_frame, canny_edge
from segmentation import tao_background_subtractor, segmentation_pipeline
from video_reader import mo_video, lay_frame, lay_thong_tin_video

# ========================
# CẤU HÌNH
# ========================
VIDEO_PATH  = os.path.join(BASE_DIR, "Data", "videos", "video3.mp4")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "best.pt")
OUTPUT_PATH = os.path.join(BASE_DIR, "Output", "result.avi")

LINE_Y      = 300       # Điều chỉnh ngưỡng đếm (từ 260 → 300)
CONF        = 0.5       # Tăng confidence YOLO (từ 0.4 → 0.5) để giảm false positive
REAL_COUNT  = 10
MAX_DISAPPEARED = 50    # Tăng từ 20 → 50 để giảm tạo ID mới

# ========================
# KHỞI TẠO
# ========================
print("[INFO] Loading YOLO model...")
model = YOLO("C:\\Users\\vinh\\runs\\detect\\train3\\weights\\best.pt")

print("[INFO] Initializing HOG detector...")
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

print("[INFO] Initializing Background Subtractor (MOG2)...")
subtractor = tao_background_subtractor()

print("[INFO] Initializing tracker...")
ct = CentroidTracker(maxDisappeared=MAX_DISAPPEARED)

print("[INFO] Opening video...")
cap = mo_video(VIDEO_PATH)
info = lay_thong_tin_video(cap)
print(f"[INFO] Video info: {info}")

delay = int(1000 / info["fps"]) if info["fps"] > 0 else 30

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

total_count = 0
counted_ids = set()

# ========================
# MAIN LOOP
# ========================
while cap.isOpened():
    frame = lay_frame(cap)
    if frame is None:
        break

    # Bước 1 — Preprocessing (Ch.2)
    gray_eq, frame_resized = preprocess_frame(frame)
    
    # Bước 1b — Edge Detection with Contours (Ch.2/Ch.3)
    edge_map, contours = canny_edge(gray_eq)

    # Khởi tạo output video theo kích thước thực
    if out is None:
        h, w = frame_resized.shape[:2]
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, info["fps"], (w, h))
        if not out.isOpened():
            print("[ERROR] Không tạo được file output!")

    h, w = frame_resized.shape[:2]

    # Bước 2 — Segmentation: MOG2 + Morphology (Ch.4)
    fg_mask, rects_mog2 = segmentation_pipeline(subtractor, frame_resized)

    # Bước 3 — Detection: HOG + SVM (Ch.5)
    boxes_hog, _ = hog.detectMultiScale(
        frame_resized,
        winStride=(8, 8),
        padding=(4, 4),
        scale=1.05
    )
    rects_hog = []
    for (x, y, bw, bh) in boxes_hog:
        rects_hog.append((x, y, x + bw, y + bh))

    # Bước 4 — Detection: YOLOv8 (Ch.5)
    results = model.predict(frame_resized, classes=[0], conf=CONF, verbose=False)
    rects_yolo = []
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.astype("int")
            rects_yolo.append((x1, y1, x2, y2))

    # Bước 5 — Kết hợp: ưu tiên YOLO, fallback MOG2 nếu YOLO không detect được
    if len(rects_yolo) > 0:
        rects_final = rects_yolo
        detect_mode = "YOLO"
    elif len(rects_mog2) > 0:
        rects_final = rects_mog2
        detect_mode = "MOG2"
    else:
        rects_final = []
        detect_mode = "None"

    # Vẽ bounding box theo mode đang dùng
    color_box = (0, 255, 0) if detect_mode == "YOLO" else (255, 0, 0)
    for (x1, y1, x2, y2) in rects_final:
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color_box, 2)

    # Vẽ HOG boxes riêng (màu vàng) để so sánh
    for (x1, y1, x2, y2) in rects_hog:
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 255), 1)

    # Bước 6 — Tracking: Centroid Tracker (Ch.4)
    objects = ct.update(rects_final)

    for (objectID, centroid) in objects.items():
        cv2.circle(frame_resized, tuple(centroid), 4, (0, 255, 0), -1)
        cv2.putText(
            frame_resized,
            f"ID:{objectID}",
            (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 2
        )

        # Bước 7 — Counting
        if centroid[1] > LINE_Y and objectID not in counted_ids:
            total_count += 1
            counted_ids.add(objectID)

    # Bước 8 — Visualization
    cv2.line(frame_resized, (0, LINE_Y), (w, LINE_Y), (0, 0, 255), 2)

    cv2.putText(
        frame_resized,
        f"Count: {total_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 0, 255), 3
    )

    overlay = frame_resized.copy()
    cv2.rectangle(overlay, (15, 60), (280, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame_resized, 0.5, 0, frame_resized)

    cv2.putText(
        frame_resized,
        f"Mode: {detect_mode}",
        (25, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (0, 255, 255), 1
    )
    cv2.putText(
        frame_resized,
        "Status: Tracking",
        (25, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (0, 255, 0), 1
    )
    cv2.putText(
        frame_resized,
        f"MAE: {abs(REAL_COUNT - total_count)}",
        (25, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 1
    )
    
    # Debug info (dùng để phân tích lỗi)
    cv2.putText(
        frame_resized,
        f"Detections: {len(rects_final)} | Tracked: {len(objects)}",
        (25, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4, (200, 200, 200), 1
    )

    # Legend
    cv2.putText(frame_resized, "YOLO", (w - 120, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame_resized, "HOG", (w - 120, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    out.write(frame_resized)
    cv2.imshow("People Tracking & Counting", frame_resized)
    cv2.imshow("MOG2 Mask", fg_mask)
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
print("\n--- KẾT QUẢ ĐÁNH GIÁ ---")
mae = abs(REAL_COUNT - total_count)
accuracy = (1 - mae / REAL_COUNT) * 100 if REAL_COUNT > 0 else 0
print(f"Số người thực tế : {REAL_COUNT}")
print(f"Số người AI đếm  : {total_count}")
print(f"Sai số MAE       : {mae}")
print(f"Độ chính xác     : {accuracy:.2f}%")
print(f"Video kết quả    : {OUTPUT_PATH}")