import numpy as np
import cv2
from planarH import computeH, warpPerspective
from matchPics import matchPicsORB


#Write script for Q8
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')