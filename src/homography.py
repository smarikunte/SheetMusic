import numpy as np


def image_to_cartesian(corners, page_width, page_height):
    X = np.array(corners)
    Y = np.array([(0, 0), (page_width, 0), (0, page_height), (page_width, page_height)])

    X_ = np.hstack([X, np.ones([X.shape[0], 1])])

    A = np.zeros([8, 9])
    A[0::2, 0:3] = X_
    A[1::2, 3:6] = X_
    A[0::2, 6:9] = X_ * -Y[:, [0]]
    A[1::2, 6:9] = X_ * -Y[:, [1]]

    _, _, V = np.linalg.svd(A)
    H = V[-1, :] / V[-1, -1]
    H = np.reshape(H, [3, 3])
    return H