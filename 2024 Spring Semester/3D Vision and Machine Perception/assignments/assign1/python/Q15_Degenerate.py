import numpy as np
import cv2
from matchPics import *
from planarH import *
# Import necessary functions

# Write script for Q15

left_img = cv2.imread('../result/bad_case_left.jpg')
right_img = cv2.imread('../result/bad_case_right.jpg')

matches, locs1, locs2 = matchPicsORB(left_img, right_img)

matches = matches[:12]

H2to1, _ = computeH_ransac(locs1[matches[:, 0]], locs2[matches[:, 1]], 200, 2)

pano_img_shape = (
    left_img.shape[0] + right_img.shape[0] * 2, left_img.shape[1] + right_img.shape[1] * 2)

warped_right_img, _ = warpPerspective(
    right_img, H2to1, (pano_img_shape[0], pano_img_shape[1]))

warped_right_img[:left_img.shape[0], :left_img.shape[1]] = left_img

# remove extra black regions
rows_to_remove = []
cols_to_remove = []
for row in range(warped_right_img.shape[0]):
    if np.all(warped_right_img[row] == 0):
        rows_to_remove.append(row)

for col in range(warped_right_img.shape[1]):
    if np.all(warped_right_img[:, col] == 0):
        cols_to_remove.append(col)

panorama_img = np.delete(warped_right_img, rows_to_remove, axis=0)
panorama_img = np.delete(panorama_img, cols_to_remove, axis=1)

cv2.imwrite("../result/bad_case_pano.jpg", panorama_img)
