import numpy as np
from src.cluster_space import ClusterSpace


def kMeans(dataPoints, maxEpochs, numClusters=1, currSpace=ClusterSpace()):
    clusterSpace = currSpace
    init = len(clusterSpace.clusters) == 0

    # kmeans++ initialization
    if init:
        randPoint = dataPoints[np.random.randint(0, dataPoints.shape[0]), :]
        clusterSpace.addCluster(randPoint)
        for i in range(numClusters - 1):
            maxDist = -np.inf
            furthestPoint = None
            for j in range(dataPoints.shape[0]):
                minDist = np.inf
                for centroid in clusterSpace.centroids():
                    dist = np.linalg.norm(centroid - dataPoints[j, :])
                    if dist < minDist:
                        minDist = dist
                if minDist > maxDist:
                    maxDist = minDist
                    furthestPoint = dataPoints[j, :]
            clusterSpace.addCluster(furthestPoint)

    epoch = 0
    prevClusters = []
    converged = False

    while epoch < maxEpochs and not converged:
        # print("Running epoch {}".format(epoch), flush=True)

        # clear cluster points, if init
        numPoints = clusterSpace.numPoints
        if init:
            for cluster in clusterSpace.clusters:
                cluster.clearPoints()

        # assign each point to nearest cluster
        for i in range(max(dataPoints.shape[0], numPoints) if init else dataPoints.shape[0]):
            # find nearest centroid
            point = dataPoints[i, :]
            distance = np.inf
            closestCluster = None
            for cluster in clusterSpace.clusters:
                newDistance = np.linalg.norm(cluster.centroid - point)
                if newDistance < distance:
                    distance = newDistance
                    closestCluster = cluster
            if closestCluster is not None:
                closestCluster.addPoint(point)

        # for each cluster generate new centroid
        for cluster in clusterSpace.clusters:
            if cluster.pointsTensor().size > 0:
                cluster.setCentroid(np.mean(cluster.pointsTensor(), 0))

        # check convergence
        for i in range(len(prevClusters)):
            converged = np.array_equal(clusterSpace.clusters[i].pointsTensor(), prevClusters[i].pointsTensor())

        # set previous clusters
        prevClusters = clusterSpace.clusters
        dataPoints = clusterSpace.pointsTensor()
        init = True
        epoch += 1

    # print("Finished")
    return clusterSpace
