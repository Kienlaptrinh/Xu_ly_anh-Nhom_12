import cv2 as cv
import numpy as np
from doc_video import mo_video, lay_frame
from preprocessing import resize_keep_ratio

# TẠO BACKGROUND SUBTRACTOR MOG2
def tao_mog2():
    return cv.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=25,
        detectShadows=False
    )

# TẠO FOREGROUND MASK
def tao_fgmask(fgbg, frame):
    return fgbg.apply(frame)

# MORPHOLOGY: ERODE + DILATE
def xu_ly_morphology(mask):
    # đưa về mask nhị phân rõ trắng đen
    _, mask = cv.threshold(mask, 200, 255, cv.THRESH_BINARY)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

    # erode giảm nhiễu trắng nhỏ
    mask = cv.erode(mask, kernel, iterations=1)

    # dilate khôi phục vùng người
    mask = cv.dilate(mask, kernel, iterations=2)

    return mask

# LỌC VÙNG NHIỄU NHỎ
def loc_nhieu(mask, min_area=40, max_area=1000):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    mask_loc = np.zeros_like(mask)

    for cnt in contours:
        area = cv.contourArea(cnt)

        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv.boundingRect(cnt)

        if h == 0:
            continue

        ratio = w / float(h)

        # lọc hình dạng không giống người (top-view)
        if ratio < 0.3 or ratio > 2.5:
            continue

        cv.drawContours(mask_loc, [cnt], -1, 255, thickness=cv.FILLED)

    return mask_loc

# TÁCH NGƯỜI KHỎI NỀN
def tach_nguoi(frame, mask):
    return cv.bitwise_and(frame, frame, mask=mask)

# PIPELINE SEGMENTATION
def segmentation_pipeline(frame, fgbg):
    # chỉ resize để đồng bộ với code trước
    frame_resized = resize_keep_ratio(frame, width=800)

    # MOG2 đúng như code mẫu của bạn
    fgmask = tao_fgmask(fgbg, frame_resized)

    # morphology theo yêu cầu
    mask_morph = xu_ly_morphology(fgmask)

    # lọc bỏ vùng nhỏ
    mask_clean = loc_nhieu(mask_morph, min_area=35)

    # tách người ra khỏi nền
    person_only = tach_nguoi(frame_resized, mask_clean)

    return frame_resized, fgmask, mask_clean, person_only

# SẮP 4 CỬA SỔ Ở 4 GÓC
def dat_4_cua_so():
    w = 640
    h = 360

    cv.namedWindow("Original", cv.WINDOW_NORMAL)
    cv.namedWindow("Foreground Mask", cv.WINDOW_NORMAL)
    cv.namedWindow("Mask Sau Morphology", cv.WINDOW_NORMAL)
    cv.namedWindow("Nguoi Tach Khoi Nen", cv.WINDOW_NORMAL)

    cv.resizeWindow("Original", w, h)
    cv.resizeWindow("Foreground Mask", w, h)
    cv.resizeWindow("Mask Sau Morphology", w, h)
    cv.resizeWindow("Nguoi Tach Khoi Nen", w, h)

    cv.moveWindow("Original", 0, 0)
    cv.moveWindow("Foreground Mask", w, 0)
    cv.moveWindow("Mask Sau Morphology", 0, h)
    cv.moveWindow("Nguoi Tach Khoi Nen", w, h)

# MAIN
if __name__ == "__main__":
    duong_dan_video = "du_lieu/video_test.mp4"

    cap = mo_video(duong_dan_video)
    fgbg = tao_mog2()

    dat_4_cua_so()

    print("Dang chay segmentation... Bam 'q' de thoat.")

    while True:
        frame = lay_frame(cap)
        if frame is None:
            print("Da phat het video.")
            break

        frame_resized, fgmask, mask_clean, person_only = segmentation_pipeline(frame, fgbg)

        cv.imshow("Original", frame_resized)
        cv.imshow("Foreground Mask", fgmask)
        cv.imshow("Mask Sau Morphology", mask_clean)
        cv.imshow("Nguoi Tach Khoi Nen", person_only)

        k = cv.waitKey(30) & 0xFF
        if k == ord('q') or k == 27:
            break

    cap.release()
    cv.destroyAllWindows()