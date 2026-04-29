import cv2
import os
import sys

def doc_anh(duong_dan_anh):
    anh = cv2.imread(duong_dan_anh)
    if anh is None:
        raise ValueError(f"Lỗi: Không tìm thấy ảnh tại '{duong_dan_anh}'!")
    return anh

def lay_thong_tin_anh(anh):
    h, w = anh.shape[:2]
    channels = anh.shape[2] if len(anh.shape) == 3 else 1
    return {"width": w, "height": h, "channels": channels}

if __name__ == "__main__":
    sys.path.append(os.path.dirname(__file__))
    from preprocessing import resize_keep_ratio

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(BASE_DIR, "data", "images", "pic1.jpg")

    anh_test = doc_anh(image_path)
    info = lay_thong_tin_anh(anh_test)
    print(f"Đọc ảnh thành công! Thông tin: {info}")

    anh_preview = resize_keep_ratio(anh_test)
    cv2.imshow('Preview Ảnh', anh_preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()