import numpy as np
import cv2
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection


def matchPics(I1, I2):
    # I1, I2 : Images to match

    # Convert Images to GrayScale
    I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    # Detect Features in Both Images
    locs1 = corner_detection(I1_gray)
    locs2 = corner_detection(I2_gray)

    # Obtain descriptors for the computed feature locations
    desc1, locs1 = computeBrief(I1_gray, locs1)
    desc2, locs2 = computeBrief(I2_gray, locs2)

    # Match features using the descriptors
    matches = briefMatch(desc1, desc2, ratio=0.8)

    return matches, locs1, locs2


def matchPicsORB(I1, I2):

    # Convert images to grayscale
    I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    key_point_1, desc_1 = orb.detectAndCompute(I1, None)
    key_point_2, desc_2 = orb.detectAndCompute(I2, None)

    # Identify the matches
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf_matcher.match(desc_1, desc_2)

    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Get the indices of the matches from the two images
    # top 8 matches (twice the minimum required matches for homography estimation)
    matches = [[m.queryIdx, m.trainIdx] for m in matches[:8]]
    matches = np.array(matches).reshape(-1, 2)

    locs1 = np.array([kp.pt for kp in key_point_1])
    locs2 = np.array([kp.pt for kp in key_point_2])

    return matches, locs1, locs2
