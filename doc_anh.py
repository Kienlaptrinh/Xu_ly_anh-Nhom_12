import cv2

def doc_anh(duong_dan_anh):
    """
    Hàm đọc ảnh từ ổ cứng và trả về dữ liệu cho các bước xử lý tiếp theo.
    """
    # Đọc ảnh bằng OpenCV
    anh = cv2.imread(duong_dan_anh)

    # Kiểm tra xem ảnh có tồn tại hay không
    if anh is None:
        print(f"Lỗi: Không tìm thấy ảnh tại '{duong_dan_anh}'!")
        return None
        
    # Trả về ma trận ảnh gốc để Nam (hoặc thành viên khác) dùng tiếp
    return anh


if __name__ == "__main__":
    duong_dan = 'du_lieu/test.jpg'
    anh_test = doc_anh(duong_dan)
    
    if anh_test is not None:
        print("Đọc ảnh thành công! Đang hiển thị... (Bấm phím bất kỳ để thoát)")
        
        
        chieu_rong_moi = 800
        chieu_cao_moi = 600
        anh_preview = cv2.resize(anh_test, (chieu_rong_moi, chieu_cao_moi))
        
        # Hiển thị bản đã thu nhỏ
        cv2.imshow('Kiem tra Input Anh (Preview)', anh_preview)
        
    
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()