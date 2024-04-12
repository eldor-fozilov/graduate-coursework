import numpy as np
import cv2
from matchPics import matchPics
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from helper import saveMatches


# Q6
# Read the image and convert to grayscale, if necessary
cv_cover = cv2.imread('../data/cv_cover.jpg')


# Initialize the histogram
hist = []

# the list of angles for showing the matched keypoints
selected_angles = [10, 90, 180, 270, 350]

for i in range(36):
    # Rotate Image
    rotated_image = rotate(cv_cover, i*10)
    # Compute features, descriptors and Match features
    matches, locs1, locs2 = matchPics(cv_cover, rotated_image)

    if i*10 in selected_angles:
        saveMatches(cv_cover, rotated_image, matches, locs1,
                    locs2, f"../result/Q6_{i*10}_rotation.png")

        # Update histogram
    hist.append(len(matches))

# Display histogram
plt.figure()
plt.bar(range(0, 360, 10), hist, width=8)
plt.xlabel('Degree')
plt.ylabel('Matching keypoints')
plt.title("BRIEF and Rotations")
plt.show()
