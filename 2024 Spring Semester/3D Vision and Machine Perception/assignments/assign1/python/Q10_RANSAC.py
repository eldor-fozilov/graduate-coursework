import numpy as np
import cv2
from matchPics import matchPicsORB
from planarH import computeH_ransac
import matplotlib.pyplot as plt

# Q10
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')

matches, locs1, locs2 = matchPicsORB(cv_desk, cv_cover)

inliers_list = []
# The number of maximum iteration
for max_iter in range(1, 241, 30):

    avg_inliers = 0

    # repeat 10 times to get average number of inliers
    for i in range(10):
        # RANSAC
        bestH2to1, inliers = computeH_ransac(
            locs1[matches[:, 0]], locs2[matches[:, 1]], max_iter, 2)
        # count inliers
        avg_inliers += np.sum(inliers)

    avg_inliers = int(avg_inliers / 10)
    inliers_list.append(avg_inliers)

# plot the number of inliers for each iteration
plt.figure()
plt.plot(list(range(1, 241, 30)), inliers_list)
plt.xlabel("Iteration")
plt.ylabel("Number of inliers")
plt.savefig("../result/Q10_figure.jpg")
plt.show()
