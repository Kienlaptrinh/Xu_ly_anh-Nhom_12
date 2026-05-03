"""
tracker.py — Centroid Tracking (Ch.4)
======================================
Chương 4 — Phân đoạn & Tracking:
  - Centroid Tracker    : gán ID duy nhất cho từng người qua các frame
  - Hungarian-style matching : ghép centroid cũ/mới bằng khoảng cách Euclidean
  - Direction detection : xác định chiều di chuyển (lên/xuống, trái/phải)
  - Counting line       : đếm người vượt qua đường kẻ (2 chiều IN/OUT)

Lý do chọn Centroid Tracking thay vì SORT/DeepSORT:
  - Đơn giản, không cần mô hình học sâu bổ sung
  - Phù hợp camera cố định, người di chuyển tương đối chậm
  - Dễ giải thích thuật toán trong báo cáo (yêu cầu Ch.4)
  - Kết hợp được với cả YOLO lẫn MOG2 segmentation
"""

import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict, deque


class CentroidTracker:
    """
    Centroid Tracking (Ch.4).

    Mỗi người được gán ID duy nhất dựa trên tâm (centroid) bounding box.
    Matching giữa các frame dùng khoảng cách Euclidean nhỏ nhất.

    Thuật toán matching (greedy, tương tự Hungarian):
      1. Tính ma trận khoảng cách D[i,j] giữa centroid cũ i và mới j
      2. Sắp xếp theo khoảng cách nhỏ nhất
      3. Gán cặp (i,j) nếu D[i,j] < maxDistance và cả i,j chưa được dùng
      4. Centroid cũ không được match → tăng disappeared counter
      5. Centroid mới không được match → đăng ký ID mới
    """

    def __init__(self, maxDisappeared=40, maxDistance=80,
                 line_y=None, line_x=None, trail_length=30):
        """
        Args:
            maxDisappeared : số frame tối đa một ID được phép mất tích
                             trước khi bị xóa. 40 frame ≈ 1.3s ở 30fps.
                             Tăng → giữ ID lâu hơn khi bị che khuất
                             Giảm → xóa nhanh hơn, ít ID ảo hơn
            maxDistance    : khoảng cách Euclidean tối đa để match (pixel).
                             Phụ thuộc vào tốc độ di chuyển và fps.
                             Công thức ước tính: max_speed_px_per_frame * 1.5
            line_y         : tọa độ Y của counting line ngang (None = tắt)
            line_x         : tọa độ X của counting line dọc (None = tắt)
            trail_length   : số frame lưu lịch sử vết đi (để vẽ trail)
        """
        self.nextObjectID   = 0
        self.objects        = OrderedDict()   # {ID: centroid (cx, cy)}
        self.disappeared    = OrderedDict()   # {ID: số frame mất tích}
        self.bboxes         = OrderedDict()   # {ID: (x1,y1,x2,y2)}
        self.trails         = OrderedDict()   # {ID: deque[(cx,cy)]}
        self.directions     = OrderedDict()   # {ID: 'up'|'down'|'left'|'right'|None}

        self.maxDisappeared = maxDisappeared
        self.maxDistance    = maxDistance
        self.trail_length   = trail_length

        # Counting line ngang (line_y)
        self.line_y         = line_y
        self.counted_ids_up   = set()   # ID đã đếm đi lên (vượt line từ dưới lên)
        self.counted_ids_down = set()   # ID đã đếm đi xuống
        self.count_in         = 0       # vào (đi xuống qua line)
        self.count_out        = 0       # ra  (đi lên qua line)

        # Counting line dọc (line_x)
        self.line_x           = line_x
        self.counted_ids_left  = set()
        self.counted_ids_right = set()
        self.count_left        = 0
        self.count_right       = 0

    # ─────────────────────────────────────────
    #  Register / Deregister
    # ─────────────────────────────────────────

    def register(self, centroid, bbox=None):
        """Đăng ký người mới với ID tiếp theo."""
        oid = self.nextObjectID
        self.objects[oid]    = centroid
        self.disappeared[oid] = 0
        self.bboxes[oid]     = bbox
        self.trails[oid]     = deque(maxlen=self.trail_length)
        self.trails[oid].append(centroid)
        self.directions[oid] = None
        self.nextObjectID   += 1

    def deregister(self, objectID):
        """Xóa người đã biến mất quá lâu."""
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.bboxes[objectID]
        del self.trails[objectID]
        del self.directions[objectID]

    # ─────────────────────────────────────────
    #  Direction Detection
    # ─────────────────────────────────────────

    def _update_direction(self, objectID, new_centroid):
        """
        Xác định chiều di chuyển dựa trên lịch sử trail.
        Dùng trung bình 5 frame gần nhất để tránh nhiễu.

        Returns: 'up' | 'down' | 'left' | 'right' | None
        """
        trail = self.trails[objectID]
        if len(trail) < 3:
            return None

        # Lấy điểm cách đây 5 frame (hoặc ít hơn nếu trail ngắn)
        lookback = min(5, len(trail))
        old_cx, old_cy = trail[-lookback]
        new_cx, new_cy = new_centroid

        dx = new_cx - old_cx
        dy = new_cy - old_cy

        # Chỉ cập nhật nếu di chuyển đủ xa (tránh jitter)
        if abs(dx) < 3 and abs(dy) < 3:
            return self.directions[objectID]

        if abs(dy) >= abs(dx):
            return "down" if dy > 0 else "up"
        else:
            return "right" if dx > 0 else "left"

    # ─────────────────────────────────────────
    #  Counting Logic
    # ─────────────────────────────────────────

    def _check_counting(self, objectID, old_centroid, new_centroid):
        """
        Kiểm tra người có vượt qua counting line không.
        Dùng old/new centroid để xác định chiều vượt (IN/OUT).

        Counting line ngang (line_y):
          - Đi xuống (old_cy < line_y ≤ new_cy) → count_in  (+1)
          - Đi lên   (old_cy > line_y ≥ new_cy) → count_out (+1)

        Counting line dọc (line_x):
          - Đi phải  (old_cx < line_x ≤ new_cx) → count_right (+1)
          - Đi trái  (old_cx > line_x ≥ new_cx) → count_left  (+1)
        """
        old_cx, old_cy = old_centroid
        new_cx, new_cy = new_centroid

        # Counting line ngang
        if self.line_y is not None:
            # Đi xuống qua line (IN)
            if (old_cy < self.line_y <= new_cy and
                    objectID not in self.counted_ids_down):
                self.counted_ids_down.add(objectID)
                self.count_in += 1

            # Đi lên qua line (OUT)
            elif (old_cy > self.line_y >= new_cy and
                  objectID not in self.counted_ids_up):
                self.counted_ids_up.add(objectID)
                self.count_out += 1

        # Counting line dọc
        if self.line_x is not None:
            if (old_cx < self.line_x <= new_cx and
                    objectID not in self.counted_ids_right):
                self.counted_ids_right.add(objectID)
                self.count_right += 1

            elif (old_cx > self.line_x >= new_cx and
                  objectID not in self.counted_ids_left):
                self.counted_ids_left.add(objectID)
                self.count_left += 1

    @property
    def totalCounted(self):
        """Tổng số người đã vượt counting line (cả 2 chiều)."""
        return self.count_in + self.count_out

    @property
    def currentCount(self):
        """Số người hiện tại đang được track."""
        return len(self.objects)

    # ─────────────────────────────────────────
    #  Core Update
    # ─────────────────────────────────────────

    def update(self, rects):
        """
        Cập nhật tracker với danh sách bounding boxes mới.

        Args:
            rects: list of (x1, y1, x2, y2)

        Returns:
            dict {objectID: (cx, cy)}
        """
        # Không có detection nào → tăng disappeared cho tất cả
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        # Tính centroids từ bounding boxes đầu vào
        inputCentroids = np.array([
            (int((x1 + x2) / 2), int((y1 + y2) / 2))
            for (x1, y1, x2, y2) in rects
        ], dtype="int")

        # Chưa có object nào → đăng ký tất cả
        if len(self.objects) == 0:
            for i, centroid in enumerate(inputCentroids):
                bbox = rects[i] if i < len(rects) else None
                self.register(centroid, bbox)
            return self.objects

        # ── Hungarian-style matching ──
        objectIDs       = list(self.objects.keys())
        objectCentroids = list(self.objects.values())

        # Ma trận khoảng cách Euclidean: D[i,j] = dist(old_i, new_j)
        D    = dist.cdist(np.array(objectCentroids), inputCentroids)
        rows = D.min(axis=1).argsort()       # sort old objects theo min dist
        cols = D.argmin(axis=1)[rows]        # new centroid gần nhất cho mỗi old

        usedRows = set()
        usedCols = set()

        for row, col in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue
            if D[row, col] > self.maxDistance:
                continue

            objectID      = objectIDs[row]
            old_centroid  = self.objects[objectID]
            new_centroid  = tuple(inputCentroids[col])

            # Cập nhật centroid, trail, direction
            self.objects[objectID]     = new_centroid
            self.bboxes[objectID]      = rects[col]
            self.disappeared[objectID] = 0
            self.trails[objectID].append(new_centroid)
            self.directions[objectID]  = self._update_direction(objectID, new_centroid)

            # Kiểm tra counting line
            self._check_counting(objectID, old_centroid, new_centroid)

            usedRows.add(row)
            usedCols.add(col)

        # Old objects không được match → tăng disappeared
        for row in set(range(D.shape[0])).difference(usedRows):
            objectID = objectIDs[row]
            self.disappeared[objectID] += 1
            if self.disappeared[objectID] > self.maxDisappeared:
                self.deregister(objectID)

        # New centroids không được match → đăng ký ID mới
        for col in set(range(D.shape[1])).difference(usedCols):
            bbox = rects[col] if col < len(rects) else None
            self.register(tuple(inputCentroids[col]), bbox)

        return self.objects

    # ─────────────────────────────────────────
    #  Visualize helpers
    # ─────────────────────────────────────────

    def ve_len_frame(self, frame):
        """
        Vẽ toàn bộ thông tin tracking lên frame:
          - Bounding box + ID + chiều di chuyển
          - Trail (vết đi) của từng người
          - Counting line (nếu có)
          - Bảng thống kê góc trên trái

        Returns:
            frame đã vẽ (không thay đổi frame gốc)
        """
        vis = frame.copy()
        h, w = vis.shape[:2]

        # ── Counting line ngang ──
        if self.line_y is not None:
            cv2.line(vis, (0, self.line_y), (w, self.line_y), (0, 255, 255), 2)
            cv2.putText(vis, f"IN:{self.count_in}  OUT:{self.count_out}",
                        (10, self.line_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ── Counting line dọc ──
        if self.line_x is not None:
            cv2.line(vis, (self.line_x, 0), (self.line_x, h), (0, 255, 255), 2)
            cv2.putText(vis, f"L:{self.count_left} R:{self.count_right}",
                        (self.line_x + 5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ── Từng object ──
        for objectID, centroid in self.objects.items():
            cx, cy = centroid
            direction = self.directions.get(objectID)
            bbox      = self.bboxes.get(objectID)

            # Màu theo chiều: xanh lá=xuống, đỏ=lên, trắng=ngang/unknown
            color = (0, 255, 0) if direction == "down" else \
                    (0, 0, 255) if direction == "up"   else \
                    (200, 200, 200)

            # Bounding box
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Centroid
            cv2.circle(vis, (cx, cy), 4, color, -1)

            # Label: ID + direction arrow
            dir_arrow = {"up": "↑", "down": "↓",
                         "left": "←", "right": "→"}.get(direction, "")
            label = f"ID{objectID} {dir_arrow}"
            cv2.putText(vis, label, (cx - 20, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Trail
            trail = self.trails.get(objectID, [])
            for i in range(1, len(trail)):
                alpha = int(255 * i / len(trail))  # fade in
                pt1 = (int(trail[i-1][0]), int(trail[i-1][1]))
                pt2 = (int(trail[i][0]),   int(trail[i][1]))
                trail_color = (0, alpha, 255 - alpha)
                cv2.line(vis, pt1, pt2, trail_color, 1)

        # ── Bảng thống kê ──
        stats = [
            f"Tracking : {self.currentCount}",
            f"Total IN : {self.count_in}",
            f"Total OUT: {self.count_out}",
        ]
        for i, text in enumerate(stats):
            cv2.putText(vis, text, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return vis

    def get_stats(self):
        """
        Trả về dict thống kê hiện tại — dùng cho evaluation.py.
        """
        return {
            "current_count" : self.currentCount,
            "total_in"      : self.count_in,
            "total_out"     : self.count_out,
            "total_counted" : self.totalCounted,
            "count_left"    : self.count_left,
            "count_right"   : self.count_right,
            "active_ids"    : list(self.objects.keys()),
        }


# ─────────────────────────────────────────────
#  Import cv2 chỉ trong visualize context
# ─────────────────────────────────────────────
try:
    import cv2
except ImportError:
    cv2 = None  # cho phép import tracker ở môi trường không có cv2


# ─────────────────────────────────────────────
#  Demo / test độc lập
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    import cv2

    sys.path.append(os.path.dirname(__file__))
    from preprocessing import preprocess_frame
    from segmentation  import tao_background_subtractor, segmentation_pipeline
    from video_reader  import mo_video, lay_frame, lay_thong_tin_video

    BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = os.path.join(BASE_DIR, "data", "videos", "video1.mp4")

    cap        = mo_video(video_path)
    info       = lay_thong_tin_video(cap)
    h_video    = info.get("height", 480)
    subtractor = tao_background_subtractor()
    tracker    = CentroidTracker(
        maxDisappeared=40,
        maxDistance=80,
        line_y=int(h_video * 0.55)   # counting line ở 55% chiều cao
    )
    delay = int(1000 / info["fps"]) if info["fps"] > 0 else 30

    print("Nhấn 'q' để thoát")

    while True:
        frame = lay_frame(cap)
        if frame is None:
            break

        gray_eq, edge_map, frame_resized = preprocess_frame(frame)
        _, rects = segmentation_pipeline(subtractor, frame_resized, edge_map)

        tracker.update(rects)
        vis = tracker.ve_len_frame(frame_resized)

        cv2.imshow("Centroid Tracker (Ch.4)", vis)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    stats = tracker.get_stats()
    print("\n=== Kết quả tracking ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    cap.release()
    cv2.destroyAllWindows()