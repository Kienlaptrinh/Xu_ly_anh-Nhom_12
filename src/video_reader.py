"""
video_reader.py — Đọc và xử lý video
======================================
Cung cấp các hàm tiện ích để đọc video, lấy thông tin,
và hỗ trợ ghi kết quả ra file video output.
"""

import cv2
import os
import sys


# ─────────────────────────────────────────────
#  Đọc video
# ─────────────────────────────────────────────

def mo_video(duong_dan_video):
    """
    Mở file video và trả về VideoCapture object.
    Raise lỗi rõ ràng nếu file không tồn tại hoặc không mở được.
    """
    if not os.path.exists(duong_dan_video):
        raise FileNotFoundError(f"Không tìm thấy file: '{duong_dan_video}'")
    cap = cv2.VideoCapture(duong_dan_video)
    if not cap.isOpened():
        raise ValueError(f"Không thể mở video: '{duong_dan_video}'")
    return cap


def lay_frame(cap):
    """
    Đọc 1 frame từ VideoCapture.
    Returns: frame (np.ndarray) hoặc None nếu hết video.
    """
    ret, frame = cap.read()
    return frame if ret else None


def lay_thong_tin_video(cap):
    """
    Lấy metadata của video.

    Returns:
        dict với các key: fps, total_frames, width, height, duration_s
    """
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return {
        "fps"          : fps,
        "total_frames" : total,
        "width"        : width,
        "height"       : height,
        "duration_s"   : round(total / fps, 2) if fps > 0 else 0
    }


def tua_ve_dau(cap):
    """Reset video về frame đầu tiên."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


def nhay_den_frame(cap, frame_index):
    """Nhảy đến frame bất kỳ theo index."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)


def lay_frame_hien_tai(cap):
    """Trả về index của frame hiện tại."""
    return int(cap.get(cv2.CAP_PROP_POS_FRAMES))


# ─────────────────────────────────────────────
#  Ghi video output
# ─────────────────────────────────────────────

def tao_video_writer(output_path, fps, width, height):
    """
    Tạo VideoWriter để lưu kết quả ra file .mp4.

    Args:
        output_path : đường dẫn file output (vd: 'output/result.mp4')
        fps         : frame rate (lấy từ lay_thong_tin_video)
        width, height: kích thước frame output

    Returns:
        cv2.VideoWriter object
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Không thể tạo VideoWriter tại: '{output_path}'")
    return writer


def ghi_frame(writer, frame):
    """Ghi 1 frame vào VideoWriter."""
    writer.write(frame)


# ─────────────────────────────────────────────
#  Tiện ích bổ sung
# ─────────────────────────────────────────────

def lay_danh_sach_video(folder):
    """
    Lấy danh sách tất cả file video (.mp4, .avi, .mov) trong folder.

    Returns:
        list đường dẫn tuyệt đối, đã sort theo tên
    """
    exts = (".mp4", ".avi", ".mov", ".mkv")
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Không phải thư mục: '{folder}'")
    files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ])
    return files


def in_thong_tin_video(info, duong_dan=""):
    """In metadata video ra console theo định dạng dễ đọc."""
    print("=" * 40)
    if duong_dan:
        print(f"  File    : {os.path.basename(duong_dan)}")
    print(f"  FPS     : {info['fps']:.2f}")
    print(f"  Frames  : {info['total_frames']}")
    print(f"  Size    : {info['width']} x {info['height']}")
    print(f"  Duration: {info['duration_s']} s")
    print("=" * 40)


# ─────────────────────────────────────────────
#  Demo / test độc lập
# ─────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.append(os.path.dirname(__file__))
    from preprocessing import resize_keep_ratio

    BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = os.path.join(BASE_DIR, "data", "videos", "video1.mp4")

    cap   = mo_video(video_path)
    info  = lay_thong_tin_video(cap)
    in_thong_tin_video(info, video_path)

    delay = int(1000 / info["fps"]) if info["fps"] > 0 else 30

    print("Nhấn 'q' để thoát | 'r' để tua về đầu | 's' lưu frame hiện tại")

    while True:
        frame = lay_frame(cap)
        if frame is None:
            print("Hết video.")
            break

        frame_idx = lay_frame_hien_tai(cap)
        preview   = resize_keep_ratio(frame, width=800)

        cv2.putText(preview, f"Frame: {frame_idx}/{info['total_frames']}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Video Preview", preview)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tua_ve_dau(cap)
            print("Đã tua về đầu.")
        elif key == ord('s'):
            save_path = f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(save_path, frame)
            print(f"Đã lưu: {save_path}")

    cap.release()
    cv2.destroyAllWindows()