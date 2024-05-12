import numpy as np
import cv2
from matchPics import matchPics
from planarH import *


# Write script for Q11
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

# resize hp_cover to cv_cover
hp_cover = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))

matches, locs1, locs2 = matchPics(cv_desk, cv_cover)

locs1 = locs1[:, [1, 0]]
locs2 = locs2[:, [1, 0]]

matched_locs1 = locs1[matches[:, 0]]
matched_locs2 = locs2[matches[:, 1]]

H2to1, _ = computeH_ransac(matched_locs1, matched_locs2, 200, 2)

composite_img = compositeH(H2to1, hp_cover, cv_desk)

cv2.imwrite('../result/Q11_HP.jpg', composite_img)
cv2.imshow('Warped Image', composite_img)
cv2.waitKey(0)
