
import cv2
from ultralytics import YOLO
import torch.nn as nn
import torch
import numpy as np

# PHẦN 1: ĐỊNH NGHĨA LỚP ADAPTIVEINPUTLAYER (TỪ IMAGE_3.PNG)
class AdaptiveInputLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Kiểm tra số lượng kênh màu của ảnh đầu vào 'x'
        # Giả sử kích thước của x là [batch_size, channels, height, width]
        channels = x.shape[1] 

        if channels == 1:
            # Nếu là ảnh trắng đen (1 kênh), nhân bản nó lên 3 lần dọc theo chiều kênh (dim=1)
            # x_new sẽ có kích thước [batch_size, 3, height, width]
            x = x.repeat(1, 3, 1, 1)
        
        elif channels == 3:
            # Nếu là ảnh màu, giữ nguyên
            pass
        else:
            raise ValueError(f"Dữ liệu đầu vào bị lỗi, số kênh = {channels}")
            
        return x

# PHẦN 2: CHƯƠNG TRÌNH CHẠY CHÍNH
# Khởi tạo các thành phần
print("Đang tải mô hình YOLO và khởi tạo camera...")

# Tải mô hình YOLO pre-trained (ví dụ: phiên bản nhẹ nhất yolov8n.pt)
model = YOLO(r"C:\Users\vinh\runs\detect\yolov8n_topdown_p23\weights\best.pt")

# Khởi tạo bộ phân đoạn nền MOG2 của OpenCV
backSub = cv2.createBackgroundSubtractorMOG2()

# Khởi tạo lớp đầu vào tùy chỉnh
adaptive_input_layer = AdaptiveInputLayer()

# Mở camera (webcam mặc định)
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Lỗi: Không thể mở camera.")
    exit()

print("Bắt đầu kiểm tra thời gian thực. Nhấn 'q' để thoát.")

while True:
    # 1. Đọc một khung hình từ camera
    ret, frame = capture.read()
    if not ret:
        break

    # 2. Tạo ảnh trắng đen MOG2 (FG Mask)
    #fgMask là một mảng numpy 1 kênh (H, W).
    fgMask = backSub.apply(frame)

    # --- PHẦN XỬ LÝ VÀ CHẠY MÔ HÌNH ---

    # A. Chạy mô hình trên ảnh MÀU gốc (để so sánh)
    # results_color = model(frame)

    # B. Chạy mô hình trên ảnh TRẮNG ĐEN MOG2 (sau khi đã xử lý sang 3 kênh)
    
    # 1. Chuyển đổi fgMask từ numpy (H, W) thành torch tensor (1, 1, H, W)
    # unsaturated_tensor_1ch shape: (1, 1, H, W)
    unsaturated_tensor_1ch = torch.from_numpy(fgMask).unsqueeze(0).unsqueeze(0).float()
    
    # 2. Áp dụng logic lớp 'AdaptiveInputLayer' để chuyển sang 3 kênh (1, 3, H, W)
    with torch.no_grad(): # Tắt tính toán gradient để tăng tốc
        # final_tensor_3ch shape: (1, 3, H, W)
        final_tensor_3ch = adaptive_input_layer(unsaturated_tensor_1ch)

    # 3. Chuyển đổi tensor 3 kênh trở lại định dạng ảnh OpenCV (RGB/BGR) cho mô hình
    # Dùng hàm np.ascontiguousarray() để sắp xếp lại bộ nhớ liền mạch
    processed_mask_img = final_tensor_3ch.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    processed_mask_img = np.ascontiguousarray(processed_mask_img) # <--- DÒNG CHỮA LỖI Ở ĐÂY
    
    # Chạy mô hình trên ảnh màu gốc và hiển thị
    results_color = model(frame)
    # cv2.imshow('Ảnh Màu Gốc - Kết Quả Quét', results_color[0].plot())

    # Chạy mô hình trên ảnh trắng đen đã được xử lý (trở thành ảnh 3 kênh)
    results_trắng_đen = model(processed_mask_img)
    # processed_mask_img hiện tại là một ảnh numpy 3 kênh (H, W, 3) với tất cả các kênh giống nhau.
    # results_trắng_đen[0].plot() sẽ vẽ khung bao lên chính ảnh này.
    # cv2.imshow hiển thị ảnh numpy 3 kênh. Vì các kênh giống nhau, nó sẽ hiển thị dưới dạng trắng đen.
    # cv2.imshow('Ảnh Trắng Đen MOG2 - Kết Quả Quét', results_trắng_đen[0].plot())

    # Trực quan hóa kết quả cho người dùng xem
    # cv2.imshow hiển thị ảnh numpy 3 kênh. Vì các kênh giống nhau, nó sẽ hiển thị dưới dạng trắng đen.
    cv2.imshow('Ảnh Trắng Đen MOG2 - Kết Quả Quét', results_trắng_đen[0].plot())

    cv2.imshow('Ảnh Màu Gốc - Kết Quả Quét', results_color[0].plot())
    
    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
capture.release()
cv2.destroyAllWindows()
print("Đã kết thúc kiểm tra.")