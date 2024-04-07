import numpy as np
import cv2
from matchPics import matchPicsORB
from planarH import *


#Write script for Q11
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')