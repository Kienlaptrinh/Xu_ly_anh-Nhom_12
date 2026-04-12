import cv2

def mo_video(duong_dan_video):
    """
    Hàm mở video từ ổ cứng.
    Tiếp tục sử dụng OpenCV để đọc video và trả về đối tượng VideoCapture cho các bước xử lý tiếp theo.
    """
    cap = cv2.VideoCapture(duong_dan_video)

    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video tại '{duong_dan_video}'!")
        return None
        
    return cap

# --- ĐOẠN CODE DƯỚI ĐÂY SẼ CHỈ CHẠY KHI BẠN TỰ TEST FILE NÀY ---
if __name__ == "__main__":
    duong_dan = 'du_lieu/video_test.mp4'
    cap_test = mo_video(duong_dan)
    
    if cap_test is not None:
        print("Mở video thành công! Đang phát... (Bấm 'q' để thoát)")
        
        while True:
            # Đọc frame nguyên bản (frame này sẽ được Nam dùng)
            ret, frame = cap_test.read() 
            
            if not ret:
                print("Đã phát hết video.")
                break
                
            # --- CÁCH SỬA LỖI VIDEO TO QUÁ MỨC ---
            # Tạo một bản sao 'frame_preview' thu nhỏ chỉ để xem trên màn hình
            chieu_rong_moi = 800
            chieu_cao_moi = 600
            frame_preview = cv2.resize(frame, (chieu_rong_moi, chieu_cao_moi))
            
            # Hiển thị bản đã thu nhỏ
            cv2.imshow('Kiem tra Input Video (Preview)', frame_preview)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
                
        cap_test.release()
        cv2.destroyAllWindows()