import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict


class CentroidTracker:
    """
    Centroid Tracking (Ch.4):
    Mỗi người được gán ID duy nhất dựa trên tâm (centroid) bounding box.
    Matching giữa các frame dùng khoảng cách Euclidean nhỏ nhất.
    """

    def __init__(self, maxDisappeared=40, maxDistance=80, line_y=None):
        """
        Args:
            maxDisappeared : số frame tối đa một ID được phép mất tích trước khi bị xóa
            maxDistance    : khoảng cách Euclidean tối đa để match 2 centroid (pixel)
            line_y         : tọa độ Y của counting line (None = không đếm)
        """
        self.nextObjectID  = 0
        self.objects       = OrderedDict()   # {ID: centroid}
        self.disappeared   = OrderedDict()   # {ID: số frame mất tích}
        self.maxDisappeared = maxDisappeared
        self.maxDistance    = maxDistance
        self.line_y         = line_y
        self.counted_ids    = set()
        self.totalCounted   = 0

    def register(self, centroid):
        self.objects[self.nextObjectID]    = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def _check_counting(self, objectID, centroid):
        if self.line_y is None:
            return
        if objectID not in self.counted_ids and centroid[1] > self.line_y:
            self.counted_ids.add(objectID)
            self.totalCounted += 1

    def update(self, rects):
        """
        Cập nhật tracker với danh sách bounding boxes mới.

        Args:
            rects: list of (x1, y1, x2, y2)

        Returns:
            dict {objectID: centroid (cx, cy)}
        """
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.array([
            (int((x1 + x2) / 2), int((y1 + y2) / 2))
            for (x1, y1, x2, y2) in rects
        ], dtype="int")

        if len(self.objects) == 0:
            for centroid in inputCentroids:
                self.register(centroid)
            return self.objects

        objectIDs       = list(self.objects.keys())
        objectCentroids = list(self.objects.values())

        # Ma trận khoảng cách Euclidean [cũ × mới]
        D    = dist.cdist(np.array(objectCentroids), inputCentroids)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        usedRows, usedCols = set(), set()
        for row, col in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue
            if D[row, col] > self.maxDistance:
                continue
            objectID                     = objectIDs[row]
            self.objects[objectID]       = inputCentroids[col]
            self.disappeared[objectID]   = 0
            self._check_counting(objectID, inputCentroids[col])
            usedRows.add(row)
            usedCols.add(col)

        for row in set(range(D.shape[0])).difference(usedRows):
            objectID = objectIDs[row]
            self.disappeared[objectID] += 1
            if self.disappeared[objectID] > self.maxDisappeared:
                self.deregister(objectID)

        for col in set(range(D.shape[1])).difference(usedCols):
            self.register(inputCentroids[col])

        return self.objects