import numpy as np


class Cluster:
    def __init__(self, centroid, parentSpace):
        self.centroid = centroid
        self.parentSpace = parentSpace
        self.points = []
        self.numPoints = 0

    def addPoint(self, point):
        self.points.append(point)
        self.numPoints += 1
        self.parentSpace.addPoint()

    def clearPoints(self):
        self.points = []
        self.parentSpace.removePoints(self.numPoints)
        self.numPoints = 0

    def setCentroid(self, point):
        self.centroid = point

    def pointsTensor(self):
        return np.array(self.points)
