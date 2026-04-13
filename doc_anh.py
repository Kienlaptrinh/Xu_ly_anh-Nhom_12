import cv2

def doc_anh(duong_dan_anh):
    """
    Đọc ảnh từ ổ cứng
    """
    anh = cv2.imread(duong_dan_anh)

    if anh is None:
        raise ValueError(f"Lỗi: Không tìm thấy ảnh tại '{duong_dan_anh}'!")

    return anh


# ========================
# RESIZE GIỮ TỈ LỆ
# ========================
def resize_keep_ratio(img, width=800):
    h, w = img.shape[:2]
    scale = width / w
    new_h = int(h * scale)

    return cv2.resize(img, (width, new_h))


# ========================
# CHUYỂN GRAY
# ========================
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ========================
# TEST
# ========================
if __name__ == "__main__":
    duong_dan = 'du_lieu/test.jpg'
    anh_test = doc_anh(duong_dan)

    print("Đọc ảnh thành công!")

    anh_preview = resize_keep_ratio(anh_test)

    cv2.imshow('Preview Anh', anh_preview)

    cv2.waitKey(0)
    cv2.destroyAllWindows()