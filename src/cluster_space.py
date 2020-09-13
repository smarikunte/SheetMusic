import numpy as np
from src.cluster import Cluster


class ClusterSpace:
    def __init__(self):
        self.clusters = []
        self.numClusters = 0
        self.numPoints = 0

    def addCluster(self, centroid):
        self.clusters.append(Cluster(centroid, self))
        self.numClusters += 1

    def addPoint(self):
        self.numPoints += 1

    def clearPoints(self):
        for cluster in self.clusters:
            cluster.clearPoints()

    def removePoints(self, numPoints):
        self.numPoints -= numPoints

    def centroids(self):
        allCentroids = []
        for cluster in self.clusters:
            allCentroids.append(cluster.centroid)
        return allCentroids

    def pointsTensor(self):
        tensorList = []
        for cluster in self.clusters:
            if cluster.pointsTensor().size > 0:
                tensorList.append(cluster.pointsTensor())
        return np.vstack(tensorList)
