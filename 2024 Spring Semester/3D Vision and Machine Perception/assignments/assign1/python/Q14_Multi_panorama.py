import numpy as np
import cv2
from planarH import *
# Import necessary functions
from yourHelperFunctions import find_best_match, build_panorama
from copy import deepcopy

# Write script for Q14
img1 = cv2.imread('../data/image1.png')
img2 = cv2.imread('../data/image2.png')
img3 = cv2.imread('../data/image3.png')
img4 = cv2.imread('../data/image4.png')


images = [img1, img2, img3, img4]

best_panoramas = None
best_total_num_inliners = -np.inf
for i in range(len(images)):

    reference_image = images[i]
    temp_images = deepcopy(images)
    temp_images.pop(i)

    total_num_inliners = 0

    panoramas = []

    for j in range(len(temp_images)):

        H, num_inliners, best_match_idx = find_best_match(
            reference_image, temp_images)

        panorama = build_panorama(
            temp_images[best_match_idx], reference_image, H)

        panoramas.append(panorama)

        reference_image = panorama
        temp_images.pop(best_match_idx)
        total_num_inliners += num_inliners

    print("Found panorama with total inliners: ", total_num_inliners)

    if total_num_inliners > best_total_num_inliners:
        best_total_num_inliners = total_num_inliners
        best_panoramas = panoramas

# Save the best panoramas
print("-" * 50)
print("Saving the panoramas with the highest total inliners: ",
      best_total_num_inliners)
for i, panorama in enumerate(best_panoramas):
    cv2.imwrite(f"../result/panorama_{i + 2}_temp.png", panorama)
