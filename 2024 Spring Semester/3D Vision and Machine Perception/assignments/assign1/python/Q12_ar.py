import numpy as np
import cv2
from loadVid import loadVid
from matchPics import matchPicsORB
from planarH import *


#Write script for Q12
cv_cover = cv2.imread('../data/cv_cover.jpg')
panda = loadVid('../data/ar_source.mov')
book = loadVid('../data/book.mov')
