import numpy as np
import cv2
from matchPics import matchPicsORB
from planarH import computeH_ransac
import matplotlib.pyplot as plt

#Q10

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')

# The number of maximum iteration
for max_iter in range(1,241,30):
	
	avg_inliers = 0
	
	# repeat 10 times to get average number of inliers
	for i in range(10):
		# RANSAC

		# count inliers
		

# visualize the number of inliers for the number of maximum iteration