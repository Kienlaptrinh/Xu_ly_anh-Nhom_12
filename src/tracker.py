import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict


class CentroidTracker:
    """
    Thuật toán Centroid Tracking:
    - Mỗi người được gán một ID duy nhất dựa trên tâm (centroid) của bounding box
    - Giữa các frame, tính khoảng cách Euclidean để match ID cũ với detection mới
    - Nếu một ID không được match sau maxDisappeared frame → xóa khỏi danh sách
    """

    def __init__(self, maxDisappeared=40, line_y=None):
        self.nextObjectID = 0
        self.objects = OrderedDict()        # {ID: centroid}
        self.disappeared = OrderedDict()    # {ID: số frame mất tích}
        self.maxDisappeared = maxDisappeared

        # Đếm người đi qua đường kẻ
        self.line_y = line_y               # Tọa độ Y của counting line
        self.counted_ids = set()           # Các ID đã được đếm
        self.totalCounted = 0              # Tổng số người đã đếm

    def register(self, centroid):
        """Đăng ký đối tượng mới với ID tiếp theo"""
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        """Xóa đối tượng khỏi danh sách theo dõi"""
        del self.objects[objectID]
        del self.disappeared[objectID]

    def _check_counting(self, objectID, centroid):
        """Đếm người nếu đi qua counting line"""
        if self.line_y is None:
            return
        if objectID not in self.counted_ids and centroid[1] > self.line_y:
            self.counted_ids.add(objectID)
            self.totalCounted += 1

    def update(self, rects):
        """
        Cập nhật tracker với danh sách bounding boxes mới.
        rects: list of (startX, startY, endX, endY)
        Trả về: dict {objectID: centroid}
        """
        # Nếu không có detection nào trong frame này
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        # Tính centroid cho mỗi bounding box mới
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cx = int((startX + endX) / 2.0)
            cy = int((startY + endY) / 2.0)
            inputCentroids[i] = (cx, cy)

        # Nếu chưa có đối tượng nào đang track → đăng ký tất cả
        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # Ma trận khoảng cách Euclidean: hàng = cũ, cột = mới
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # Sắp xếp theo khoảng cách nhỏ nhất để ưu tiên match gần nhất
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                # Match ID cũ với centroid mới gần nhất
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                self._check_counting(objectID, inputCentroids[col])

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(D.shape[0])).difference(usedRows)
            unusedCols = set(range(D.shape[1])).difference(usedCols)

            # ID cũ không match → tăng biến mất
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # Detection mới không match → đăng ký ID mới
            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects