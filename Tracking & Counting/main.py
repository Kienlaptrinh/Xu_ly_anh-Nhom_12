import cv2
from ultralytics import YOLO
from tracker import CentroidTracker  # Import file 

# 1. Khởi tạo (Dùng model mặc định để test trên nhánh)
model = YOLO('yolov8n.pt') 
ct = CentroidTracker(maxDisappeared=20)

# 2. Cấu hình đếm
video_path = "D:\\gitclone\\Xu_ly_anh-Nhom_12\\Du_lieu\\video_test.mp4" # Thay bằng đường dẫn video của bạn
cap = cv2.VideoCapture(video_path)
line_y = 360  # Vị trí vạch đếm (có thể điều chỉnh)
total_count = 0
counted_ids = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.resize(frame, (800, 500)) # Resize để dễ quan sát

    # 3. Chạy Detection (Giả lập phần của Vinh)
    results = model.predict(frame, classes=[0], conf=0.4, verbose=False)
    rects = []
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        for box in boxes:
            rects.append(box.astype("int"))
            # Vẽ bounding box (Nhiệm vụ Visualization)
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)

    # 4. Chạy Tracking 
    objects = ct.update(rects)

    for (objectID, centroid) in objects.items():
        # Vẽ tâm và ID
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        cv2.putText(frame, f"ID:{objectID}", (centroid[0]-10, centroid[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 5. Logic Đếm người qua vạch
        if centroid[1] > line_y and objectID not in counted_ids:
            total_count += 1
            counted_ids.add(objectID)

    # 6. Hiển thị (Nhiệm vụ Visualization)
    cv2.line(frame, (0, line_y), (800, line_y), (0, 0, 255), 2)
    cv2.putText(frame, f"Count: {total_count}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Tracking & Counting", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()