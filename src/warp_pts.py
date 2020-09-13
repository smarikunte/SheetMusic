from src import homography
import numpy as np


def warp(corners, page_width, page_height, pos):
    # Warped points
    H = homography.image_to_cartesian(corners, page_width, page_height)
    target_pos = np.array(pos)
    interior_pts = np.hstack([target_pos, np.ones([target_pos.shape[0], 1])])
    warped_pts = interior_pts.dot(H.T)
    warped_pts = warped_pts[:, :2] / warped_pts[:, [2]]
    return warped_pts
