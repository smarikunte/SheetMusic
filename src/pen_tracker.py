import numpy as np
import cv2
import imutils
from src.warp_pts import warp
from src.kmeans import kMeans
from src.audio import AudioPlayer

tracker = None
cannyLowerThreshold = 200
page_width = 632
page_height = 900

BB = None

cap = cv2.VideoCapture('blue_cropped.mp4')
audioPlayer = AudioPlayer()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Frame error, exiting...")
        break
    frame = imutils.resize(frame, width=500)
    height, width = frame.shape[:2]
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey = cv2.bilateralFilter(grey, 10, 20, 20)
    edges = cv2.Canny(grey, cannyLowerThreshold, cannyLowerThreshold * 3)
    cv2.imshow("Edges", edges)
    lines = cv2.HoughLines(edges, 1, np.pi/125, 90, None, 0, 0)
    line_vectors = []
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 2000 * -b)
        y1 = int(y0 + 2000 * a)
        x2 = int(x0 - 2000 * -b)
        y2 = int(y0 - 2000 * a)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        p1 = np.array([x1, y1, 1])
        p2 = np.array([x2, y2, 1])
        l_p = np.cross(p1, p2).reshape((3, 1))
        ln = np.array([l_p[0][0] / l_p[2][0], l_p[1][0] / l_p[2][0], 1]).T
        line_vectors.append(ln)
    points = []
    for i in range(len(line_vectors)):
        for j in range(len(line_vectors)):
            if i != j:
                l1 = line_vectors[i]
                l2 = line_vectors[j]
                if abs(np.dot(l1[:2], l2[:2]) / (np.linalg.norm(l2[:2]) ** 2)) > 0.5:
                    continue
                pt_p = np.cross(l1, l2).reshape((3, 1))
                if pt_p[2][0] != 0:
                    pt = np.array([pt_p[0][0] / pt_p[2][0], pt_p[1][0] / pt_p[2][0], 1]).T
                    points.append(pt)
    points = np.array(points)[:, :2]
    points = np.array(list(filter(
        lambda c: 0 < int(c[0]) < width and 0 < int(c[1]) < height, points))
    )

    # PRE CLUSTERING POINTS VISUALIZATION
    # for point in points:
    #     x = int(point[0])
    #     y = int(point[1])
    #     cv2.circle(frame, (x, y), 3, (255, 0, 0), 3)

    # clusterSpace = gMeans(points, 1, maxClusters=5)
    clusterSpace = kMeans(points, 1, numClusters=4)

    clusterList = clusterSpace.clusters
    clusterList.sort(key=lambda c: c.numPoints, reverse=True)
    clusters = list(filter(
        lambda c: 0 < int(c.centroid[0]) < width and 0 < int(c.centroid[1]) < height, clusterList)
    )
    corners = []
    for i in range(4):
        if i < len(clusters):
            x = int(clusters[i].centroid[0])
            y = int(clusters[i].centroid[1])
            cv2.circle(frame, (x, y), 3, (255, 0, 0), 3)
            corners.append(clusters[i].centroid)

    cartesian_plane = np.ones((page_height + 5, page_width + 5, 3))

    if BB is not None:
        (success, box) = tracker.update(frame)
        if success:
            print('success')
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
            if len(corners) == 4:
                out_corners = []
                corners.sort(key=lambda c: c[1])
                top_corners = corners[: 2]
                bot_corners = corners[-2:]
                top_corners.sort(key=lambda c: c[0])
                bot_corners.sort(key=lambda c: c[0])
                out_corners.extend(top_corners)
                out_corners.extend(bot_corners)
                warped_pts = warp(out_corners, page_width, page_height, corners)
                pen_pos = warp(out_corners, page_width, page_height, [np.array([x + w/2, y + h/2])])
                pen_pos = pen_pos[0]
                # OPENCV BUILT IN ALTERNATIVE
                # image = np.float32(out_corners)
                # cartesian = np.float32([[0, 0], [632, 0], [0, 900], [632, 900]])
                # points = np.array([image])
                # H, _ = cv2.findHomography(image, cartesian)
                # warped_pts = cv2.perspectiveTransform(points, H)
                # warped_pts = np.squeeze(warped_pts, axis=0)

                # Plot corners of page
                for point in warped_pts:
                    cv2.circle(cartesian_plane, (int(point[0]), int(point[1])), 3, (255, 0, 0), 3)
                cv2.line(cartesian_plane, (page_width//2, 0), (page_width//2, page_height), (0, 255, 0), thickness=3)
                cv2.line(cartesian_plane, (0, page_height//2), (page_width, page_height//2), (0, 255, 0), thickness=3)
                audioPlayer.play_sound(pen_pos, page_width, page_height)
                cv2.circle(cartesian_plane, (int(pen_pos[0]), int(pen_pos[1])), 3, (255, 0, 0), 3)
        else:
            print('fail')

    cv2.imshow("cart", cartesian_plane)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Press S to set bounding box, Q to quit
    if key == ord("s"):
        BB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, BB)
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
