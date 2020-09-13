import matplotlib.pyplot as plt
import numpy as np


def clusterVis(clusterSpace):
    fig, ax = plt.subplots()
    for cluster in clusterSpace.clusters:
        clr = np.random.rand(1, 3)
        ax.scatter(cluster.centroid[0], cluster.centroid[1], marker="^", c=clr)
        ax.scatter(cluster.pointsTensor()[:, 0], cluster.pointsTensor()[:, 1], c=clr)
    plt.show()

