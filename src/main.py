"""
main.py — Pipeline chính: Phát hiện và đếm người
==================================================
Entry point của hệ thống. Tích hợp toàn bộ pipeline:

  Ch.2 — preprocessing.py  : resize, Gaussian blur, CLAHE, Canny edge
  Ch.3 — preprocessing.py  : HOG descriptor + SVM detector
  Ch.4 — segmentation.py   : MOG2 background subtraction + Morphology
         tracker.py        : Centroid Tracking + counting line
  Ch.5 — YOLO detection    : YOLOv8n / best.pt

Fusion strategy: Fallback chain YOLO → HOG → MOG2
  - YOLO: accuracy cao nhất, dùng làm primary
  - HOG : backup khi YOLO không detect được (người nhỏ, xa)
  - MOG2: last resort + minh họa kỹ thuật Ch.4

Chạy:
  python src/main.py                          # video mặc định
  python src/main.py --video data/videos/video2.mp4
  python src/main.py --video data/videos/video1.mp4 --real_count 50 --conf 0.35
  python src/main.py --image data/images/pic1.jpg
  python src/main.py --all                    # chạy tất cả video trong data/videos/
"""

import cv2
import os
import sys
import argparse
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from ultralytics  import YOLO
from tracker      import CentroidTracker
from preprocessing import preprocess_frame
from segmentation  import tao_background_subtractor, segmentation_pipeline
from video_reader  import (mo_video, lay_frame, lay_thong_tin_video,
                            tao_video_writer, ghi_frame, lay_danh_sach_video)
from image_reader  import doc_anh, xu_ly_anh_tinh


# ─────────────────────────────────────────────
#  Cấu hình mặc định
# ─────────────────────────────────────────────

DEFAULT_VIDEO      = os.path.join(BASE_DIR, "data", "videos", "video1.mp4")
DEFAULT_IMAGE      = os.path.join(BASE_DIR, "data", "images", "pic1.jpg")
MODEL_PATH         = os.path.join(BASE_DIR, "models", "best.pt")
MODEL_PATH_NANO    = os.path.join(BASE_DIR, "models", "yolov8n.pt")
OUTPUT_DIR         = os.path.join(BASE_DIR, "output")

# Ground truth cho từng video (thay bằng giá trị thực tế khi có)
GROUND_TRUTH = {
    "video1.mp4": 50,
    "video2.mp4": 0,
    "video3.mp4": 0,
    "video4.mp4": 0,
}

COLOR_MAP = {
    "YOLO": (0, 255, 0),    # xanh lá
    "HOG" : (0, 165, 255),  # cam
    "MOG2": (255, 0, 0),    # xanh lam
    "None": (128, 128, 128) # xám
}


# ─────────────────────────────────────────────
#  Argparse
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Hệ thống phát hiện và đếm người — Nhóm 12"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--video", type=str, default=DEFAULT_VIDEO,
                       help="Đường dẫn file video (mặc định: video1.mp4)")
    group.add_argument("--image", type=str, default=None,
                       help="Đường dẫn ảnh tĩnh (chạy chế độ ảnh)")
    group.add_argument("--all",   action="store_true",
                       help="Chạy tất cả video trong data/videos/")

    parser.add_argument("--model",      type=str, default=MODEL_PATH,
                        help="Đường dẫn YOLO model (.pt)")
    parser.add_argument("--conf",       type=float, default=0.35,
                        help="YOLO confidence threshold (default: 0.35)")
    parser.add_argument("--line_y",     type=int,   default=None,
                        help="Tọa độ Y counting line (default: 55% chiều cao)")
    parser.add_argument("--real_count", type=int,   default=None,
                        help="Số người thực tế (ground truth) để tính metrics")
    parser.add_argument("--no_display", action="store_true",
                        help="Không hiển thị cửa sổ (chạy headless)")
    parser.add_argument("--save",       action="store_true", default=True,
                        help="Lưu video kết quả (mặc định: True)")
    return parser.parse_args()


# ─────────────────────────────────────────────
#  Khởi tạo models
# ─────────────────────────────────────────────

def khoi_tao_models(model_path):
    """Khởi tạo YOLO, HOG, MOG2. Fallback sang yolov8n nếu best.pt không có."""
    # YOLO
    if not os.path.exists(model_path):
        print(f"[WARN] Không tìm thấy {model_path}, dùng yolov8n.pt")
        model_path = MODEL_PATH_NANO
    print(f"[INFO] Loading YOLO: {os.path.basename(model_path)}")
    model = YOLO(model_path)

    # HOG + SVM (Ch.3)
    print("[INFO] Initializing HOG detector (Ch.3)...")
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # MOG2 (Ch.4)
    print("[INFO] Initializing MOG2 subtractor (Ch.4)...")
    subtractor = tao_background_subtractor(history=500, var_threshold=50)

    return model, hog, subtractor


# ─────────────────────────────────────────────
#  Fusion — kết hợp kết quả các phương pháp
# ─────────────────────────────────────────────

def fusion_detect(model, hog, subtractor, frame_resized,
                  gray_eq, edge_map, conf):
    """
    Chạy 3 phương pháp và áp fallback chain YOLO → HOG → MOG2.

    Returns:
        rects_final : list (x1,y1,x2,y2) kết quả cuối
        detect_mode : 'YOLO' | 'HOG' | 'MOG2' | 'None'
        fg_mask     : MOG2 mask (visualize)
        all_counts  : dict số lượng từng phương pháp
    """
    # Ch.4 — MOG2 + Morphology + edge-guided
    fg_mask, rects_mog2 = segmentation_pipeline(
        subtractor, frame_resized, edge_map
    )

    # Ch.3 — HOG + SVM
    boxes_hog, _ = hog.detectMultiScale(
        gray_eq, winStride=(8, 8), padding=(4, 4), scale=1.05
    )
    rects_hog = [(x, y, x + bw, y + bh) for (x, y, bw, bh) in boxes_hog]

    # Ch.5 — YOLO
    results    = model.predict(frame_resized, classes=[0],
                               conf=conf, iou=0.4, verbose=False)
    rects_yolo = [
        tuple(box.astype("int"))
        for r in results
        for box in r.boxes.xyxy.cpu().numpy()
    ]

    # Fallback chain
    if len(rects_yolo) > 0:
        rects_final, detect_mode = rects_yolo, "YOLO"
    elif len(rects_hog) > 0:
        rects_final, detect_mode = rects_hog,  "HOG"
    elif len(rects_mog2) > 0:
        rects_final, detect_mode = rects_mog2, "MOG2"
    else:
        rects_final, detect_mode = [],          "None"

    all_counts = {
        "YOLO": len(rects_yolo),
        "HOG" : len(rects_hog),
        "MOG2": len(rects_mog2),
    }

    return rects_final, detect_mode, fg_mask, all_counts


# ─────────────────────────────────────────────
#  Visualize
# ─────────────────────────────────────────────

def ve_hud(frame, ct, detect_mode, all_counts, real_count=None):
    """
    Vẽ HUD (Heads-Up Display) lên frame:
      - Counting line + IN/OUT
      - Bảng thống kê (mode, count, MAE nếu có ground truth)
      - Legend màu sắc
    """
    h, w = frame.shape[:2]
    color = COLOR_MAP[detect_mode]

    # Counting line
    if ct.line_y is not None:
        cv2.line(frame, (0, ct.line_y), (w, ct.line_y), (0, 255, 255), 2)
        cv2.putText(frame,
                    f"IN: {ct.count_in}  OUT: {ct.count_out}",
                    (10, ct.line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # Panel thông tin (nền mờ)
    panel_h = 145 if real_count is not None else 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (295, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # Thông tin
    lines = [
        (f"Tracking : {ct.currentCount} nguoi",     (20, 35),  (255, 255, 255)),
        (f"Total    : {ct.totalCounted}",            (20, 58),  (0, 255, 255)),
        (f"Mode     : {detect_mode}",                (20, 81),  color),
        (f"YOLO:{all_counts['YOLO']} HOG:{all_counts['HOG']} MOG2:{all_counts['MOG2']}",
                                                     (20, 104), (180, 180, 180)),
    ]
    if real_count is not None:
        mae = abs(real_count - ct.totalCounted)
        acc = max(0.0, (1 - mae / real_count) * 100) if real_count > 0 else 0
        lines.append((f"MAE:{mae}  Acc:{acc:.1f}%", (20, 127), (0, 200, 100)))

    for text, pos, c in lines:
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.52, c, 1)

    return frame


# ─────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────

def tinh_metrics(real_count, predicted_count):
    """
    Tính các chỉ số đánh giá định lượng cho bài toán đếm người.

    Cách tính TP/FP/FN cho bài toán đếm (không phải detection per-frame):
      TP = min(predicted, real)      — số người đếm đúng
      FP = max(0, predicted - real)  — đếm dư
      FN = max(0, real - predicted)  — bỏ sót

    Returns:
        dict chứa MAE, accuracy, precision, recall, f1
    """
    mae      = abs(real_count - predicted_count)
    accuracy = max(0.0, (1 - mae / real_count) * 100) if real_count > 0 else 0.0

    tp = min(predicted_count, real_count)
    fp = max(0, predicted_count - real_count)
    fn = max(0, real_count - predicted_count)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "real_count"      : real_count,
        "predicted_count" : predicted_count,
        "mae"             : mae,
        "accuracy_pct"    : round(accuracy, 2),
        "precision"       : round(precision, 4),
        "recall"          : round(recall, 4),
        "f1"              : round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }


def in_ket_qua(metrics, video_name="", elapsed=None):
    """In bảng kết quả đánh giá ra console."""
    print("\n" + "=" * 45)
    if video_name:
        print(f"  Video          : {video_name}")
    if elapsed:
        print(f"  Thời gian chạy : {elapsed:.1f}s")
    print(f"  Số người thực  : {metrics['real_count']}")
    print(f"  Số người đếm   : {metrics['predicted_count']}")
    print(f"  MAE            : {metrics['mae']}")
    print(f"  Accuracy       : {metrics['accuracy_pct']}%")
    print(f"  Precision      : {metrics['precision']}")
    print(f"  Recall         : {metrics['recall']}")
    print(f"  F1-score       : {metrics['f1']}")
    print("=" * 45)


# ─────────────────────────────────────────────
#  Xử lý 1 video
# ─────────────────────────────────────────────

def chay_video(video_path, model, hog, subtractor, args):
    """
    Chạy pipeline đầy đủ trên 1 file video.

    Returns:
        metrics dict (hoặc None nếu không có ground truth)
    """
    video_name = os.path.basename(video_path)
    real_count = args.real_count or GROUND_TRUTH.get(video_name, 0)

    print(f"\n[INFO] Xử lý: {video_name} | GT={real_count} | conf={args.conf}")

    cap  = mo_video(video_path)
    info = lay_thong_tin_video(cap)

    # Counting line: dùng args hoặc tự tính 55% chiều cao
    line_y = args.line_y if args.line_y else int(info["height"] * 0.55)

    # Reset subtractor cho mỗi video
    subtractor_v = tao_background_subtractor()

    ct = CentroidTracker(
        maxDisappeared=30,
        maxDistance=80,
        line_y=line_y
    )

    # VideoWriter
    writer     = None
    output_path = None
    if args.save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(
            OUTPUT_DIR, f"result_{os.path.splitext(video_name)[0]}.mp4"
        )

    delay      = int(1000 / info["fps"]) if info["fps"] > 0 else 30
    frame_idx  = 0
    t_start    = time.time()

    while True:
        frame = lay_frame(cap)
        if frame is None:
            break

        # Ch.2 — Preprocessing
        gray_eq, edge_map, frame_resized = preprocess_frame(frame)

        # Khởi tạo writer sau khi biết kích thước thực
        if writer is None and args.save and output_path:
            h_r, w_r = frame_resized.shape[:2]
            writer = tao_video_writer(output_path, info["fps"], w_r, h_r)

        # Ch.3 + Ch.4 + Ch.5 — Fusion detect
        rects_final, detect_mode, fg_mask, all_counts = fusion_detect(
            model, hog, subtractor_v,
            frame_resized, gray_eq, edge_map, args.conf
        )

        # Ch.4 — Centroid Tracking
        ct.update(rects_final)

        # Visualize
        color = COLOR_MAP[detect_mode]
        for (x1, y1, x2, y2) in rects_final:
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)

        # Vẽ trail + ID từ tracker
        frame_resized = ct.ve_len_frame(frame_resized)

        # HUD
        ve_hud(frame_resized, ct, detect_mode, all_counts,
               real_count if real_count > 0 else None)

        # Ghi output
        if writer is not None:
            ghi_frame(writer, frame_resized)

        # Hiển thị
        if not args.no_display:
            cv2.imshow(f"Tracking — {video_name}", frame_resized)
            cv2.imshow("MOG2 Mask (Ch.4)",   fg_mask)
            cv2.imshow("Canny Edges (Ch.2)", edge_map)

            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                print("[INFO] Dừng sớm.")
                break
            elif key == ord('p'):
                cv2.waitKey(0)  # pause

        frame_idx += 1

    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start

    # Evaluation
    stats = ct.get_stats()
    print(f"[INFO] Frames xử lý: {frame_idx} | "
          f"FPS thực: {frame_idx/elapsed:.1f} | "
          f"Thời gian: {elapsed:.1f}s")

    if real_count > 0:
        metrics = tinh_metrics(real_count, ct.totalCounted)
        in_ket_qua(metrics, video_name, elapsed)
        if output_path:
            print(f"[INFO] Đã lưu: {output_path}")
        return metrics
    else:
        print(f"[INFO] Tổng đếm: {ct.totalCounted} | "
              f"IN: {ct.count_in} | OUT: {ct.count_out}")
        return None


# ─────────────────────────────────────────────
#  Xử lý ảnh tĩnh
# ─────────────────────────────────────────────

def chay_anh(image_path, model, hog, subtractor, args):
    """Chạy pipeline phát hiện người trên ảnh tĩnh."""
    print(f"[INFO] Xử lý ảnh: {os.path.basename(image_path)}")

    anh = doc_anh(image_path)
    output, rects, mode, edge_map, fg_mask, all_results = xu_ly_anh_tinh(
        anh, model, hog, subtractor, conf=args.conf
    )

    print(f"[RESULT] Phát hiện: {len(rects)} người | Mode: {mode}")
    print(f"  YOLO: {len(all_results['YOLO'])} | "
          f"HOG: {len(all_results['HOG'])} | "
          f"MOG2: {len(all_results['MOG2'])}")

    if args.save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR,
            f"result_{os.path.splitext(os.path.basename(image_path))[0]}.jpg")
        cv2.imwrite(save_path, output)
        print(f"[INFO] Đã lưu: {save_path}")

    if not args.no_display:
        cv2.imshow("Detection Result", output)
        cv2.imshow("Canny Edges (Ch.2)", edge_map)
        cv2.imshow("MOG2 Mask (Ch.4)", fg_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────
#  Tổng hợp kết quả nhiều video
# ─────────────────────────────────────────────

def in_tong_hop(all_metrics):
    """In bảng tổng hợp kết quả tất cả video."""
    if not all_metrics:
        return
    valid = [m for m in all_metrics if m is not None]
    if not valid:
        return

    avg_mae  = sum(m["mae"] for m in valid) / len(valid)
    avg_acc  = sum(m["accuracy_pct"] for m in valid) / len(valid)
    avg_f1   = sum(m["f1"] for m in valid) / len(valid)

    print("\n" + "=" * 45)
    print("  TỔNG HỢP TẤT CẢ VIDEO")
    print(f"  Số video đánh giá : {len(valid)}")
    print(f"  MAE trung bình    : {avg_mae:.2f}")
    print(f"  Accuracy TB       : {avg_acc:.2f}%")
    print(f"  F1 trung bình     : {avg_f1:.4f}")
    print("=" * 45)


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    # Khởi tạo models (dùng chung cho tất cả video)
    model, hog, subtractor = khoi_tao_models(args.model)

    # ── Chế độ ảnh tĩnh ──
    if args.image:
        chay_anh(args.image, model, hog, subtractor, args)
        return

    # ── Chạy tất cả video ──
    if args.all:
        video_folder = os.path.join(BASE_DIR, "data", "videos")
        video_list   = lay_danh_sach_video(video_folder)
        if not video_list:
            print(f"[ERROR] Không tìm thấy video trong: {video_folder}")
            return
        print(f"[INFO] Tìm thấy {len(video_list)} video: "
              f"{[os.path.basename(v) for v in video_list]}")

        all_metrics = []
        for vp in video_list:
            m = chay_video(vp, model, hog, subtractor, args)
            all_metrics.append(m)

        in_tong_hop(all_metrics)
        return

    # ── Chạy 1 video ──
    chay_video(args.video, model, hog, subtractor, args)


if __name__ == "__main__":
    main()