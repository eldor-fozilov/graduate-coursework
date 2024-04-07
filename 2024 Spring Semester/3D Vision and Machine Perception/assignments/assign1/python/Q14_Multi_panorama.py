import numpy as np
import cv2
from matchPics import matchPicsORB
from planarH import *
#Import necessary functions


#Write script for Q14
img1 = cv2.imread('../data/image1.png')
img2 = cv2.imread('../data/image2.png')
img3 = cv2.imread('../data/image3.png')
img4 = cv2.imread('../data/image4.png')

# First, find the order of the given images



# Create a panorama with two neighbor images and save the panorama

cv2.imwrite('../result/panorama_2.png', )


# Create a panorama with three images and save the image

cv2.imwrite('../result/panorama_3.png', )


# Create a final panorama and save the result

cv2.imwrite('../result/panorama_4.png', )