from calendar import c
import numpy as np
import cv2
from planarH import computeH, warpPerspective
from matchPics import matchPicsORB

from helper import plotMatches

# Write script for Q8
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')

matches, locs1, locs2 = matchPicsORB(cv_cover, cv_desk)

matched_locs1 = locs1[matches[:, 0]]
matched_locs2 = locs2[matches[:, 1]]

H, W = cv_desk.shape[:2]
H2to1 = computeH(matched_locs1, matched_locs2)

warped_image, mask = warpPerspective(cv_cover, H2to1, (H, W))

# use the mask
warped_image[mask == 0] = cv_desk[mask == 0]

# save the image and then show it
cv2.imwrite('../result/Q8_output.jpg', warped_image)
cv2.imshow('Warped Image', warped_image)
cv2.waitKey(0)
