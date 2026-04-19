import cv2
from ultralytics import YOLO

# 1. Load mô hình YOLO của bạn
#model = YOLO('yolov8x.pt')
model = YOLO(r"C:\Users\vinh\runs\detect\train3\weights\best.pt")
#model = YOLO(r"C:\Users\vinh\runs\detect\yolov8n_topdown_p23\weights\best.pt")
backSub = cv2.createBackgroundSubtractorMOG2()

# 2. Mở file video .mp4
video_path = "du_lieu/istockphoto-1205908649-640_adpp_is.mp4"  # Thay đổi đường dẫn đến video của bạn
cap = cv2.VideoCapture(video_path)


# Kiểm tra xem có mở được video không
if not cap.isOpened():
    print("Lỗi: Không thể mở file video!")
    exit()

while True:
    # Đọc từng khung hình (frame) từ video
    ret, frame = cap.read()
    
    # Nếu không đọc được frame nữa (hết video) thì thoát vòng lặp
    if not ret:
        break


    framemask = backSub.apply(frame)

    #mask_3ch = cv2.cvtColor(framemask, cv2.COLOR_GRAY2BGR)
    mask_3ch = frame.copy()

    # 3. Đưa khung hình vào YOLO để nhận diện
    # stream=True giúp tiết kiệm bộ nhớ khi xử lý video dài
    results = model.track(mask_3ch, classes=[0], conf=0.3, iou=0.7, imgsz=1280, persist=True, tracker="bytetrack.yaml", verbose=False)

    # 4. Vẽ kết quả lên khung hình
    for r in results:
        annotated_frame = mask_3ch.copy()
        xyxy = r.boxes.xyxy.cpu().numpy()  # Lấy tọa độ hộp bao
        confs = r.boxes.conf.cpu().numpy().flatten()  # Lấy độ tin cậy
        for (x1, y1, x2, y2), conf in zip(xyxy, confs):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_frame,
                f'Nguoi: {conf:.2f}',
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        # 5. Hiển thị khung hình đã được vẽ kết quả
        cv2.imshow("YOLOv8 Detection - Video", cv2.resize(annotated_frame, (640, 640)))
        
    # Nhấn phím 'q' trên bàn phím để thoát video sớm
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
