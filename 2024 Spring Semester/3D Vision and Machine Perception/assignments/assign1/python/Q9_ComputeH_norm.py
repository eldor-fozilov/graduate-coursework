from matplotlib.pyplot import gray
import numpy as np
import cv2
from planarH import computeH, computeH_norm, warpPerspective
from matchPics import matchPicsORB
# Import necessary functions


# Write script for Q9
I1 = cv2.imread('../data/book_small_scale.jpg')
I2 = cv2.imread('../data/book_big_scale.jpg')

matches, locs1, locs2 = matchPicsORB(I1, I2)

# take only the top closest 8 (hyperparameter) matches
matches = matches[:8]

matched_locs1 = locs1[matches[:, 0]]
matched_locs2 = locs2[matches[:, 1]]

H, W = I1.shape[:2]
H2to1 = computeH_norm(matched_locs1, matched_locs2)

warped_image, mask = warpPerspective(I2, H2to1, (H, W))

# use the mask
warped_image[mask == 0] = I1[mask == 0]

# save the image and then show it
cv2.imwrite('../result/Q9_output.jpg', warped_image)
cv2.imshow('Warped Image', warped_image)
cv2.waitKey(0)
