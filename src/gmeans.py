import numpy as np
from src.kmeans import kMeans
from src.anderson import anderson
from src.cluster_space import ClusterSpace
from src.cluster_vis import clusterVis


def gMeans(dataPoints, maxEpochs, numClusters=1, currSpace=ClusterSpace(), sigLevel=0.05, maxClusters=250):

    # ABSOLUTE FUDGE LORD
    # sigLevel /= dataPoints.shape[1] ** np.e ** 2.56
    # print(dataPoints.shape)
    clusterSpace = kMeans(dataPoints, maxEpochs, numClusters, currSpace)
    clusterSplit = True
    while clusterSplit and clusterSpace.numClusters < maxClusters:
        clusterSplit = False
        for cluster in clusterSpace.clusters:
            if cluster.pointsTensor().shape[0] > 1:
                cov = np.cov(cluster.pointsTensor(), rowvar=False)
                w, v = np.linalg.eig(cov)
                maxEig = np.amax(w)
                firstPC = v[:, np.where(w == maxEig)]
                firstPC = firstPC.T
                m = np.squeeze(firstPC * np.sqrt(2 * maxEig / np.pi))
                c1 = cluster.centroid + m
                c2 = cluster.centroid - m
                v = c1 - c2
                if np.linalg.norm(v) == 0:
                    continue
                x_prime = (np.dot(cluster.pointsTensor(), v) / np.linalg.norm(v))
                x_prime = (x_prime - np.mean(x_prime)) / np.std(x_prime)
                p = anderson(x_prime)
                if p < sigLevel:
                    cluster.setCentroid(c1)
                    clusterSpace.addCluster(c2)
                    clusterSplit = True
        if clusterSplit:
            clusterSpace.clearPoints()
            clusterSpace = kMeans(dataPoints, maxEpochs, clusterSpace.numClusters, clusterSpace)
        print(clusterSpace.numClusters)
    return clusterSpace


if __name__ == '__main__':
    data = np.loadtxt('a2.txt')
    clusterSpace = gMeans(data, 1, maxClusters=10)
    # Visualize
    clusterVis(clusterSpace)
