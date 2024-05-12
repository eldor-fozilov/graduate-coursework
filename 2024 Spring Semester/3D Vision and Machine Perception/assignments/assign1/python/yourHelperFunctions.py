import numpy as np
import cv2
from matchPics import matchPics, matchPicsORB
from planarH import *


def center_crop(img, new_height, new_width):
    height, width = img.shape[:2]
    start_height = height // 2 - new_height // 2
    start_width = width // 2 - new_width // 2
    return img[start_height:start_height + new_height, start_width:start_width + new_width]


def compose_image(source_image, panda_frame, book_frame):

    h, w = panda_frame.shape[:2]
    matches, locs1, locs2 = matchPics(book_frame, source_image)

    locs1 = locs1[:, [1, 0]]
    locs2 = locs2[:, [1, 0]]

    H2to1, _ = computeH_ransac(
        locs1[matches[:, 0]], locs2[matches[:, 1]], 500, 2)

    cropped_panda_frame = center_crop(panda_frame, h * 3 // 4, w * 2 // 3)
    resized_panda_frame = cv2.resize(cropped_panda_frame, dsize=(
        source_image.shape[1], source_image.shape[0]), interpolation=cv2.INTER_LINEAR)
    composed_img = compositeH(H2to1, resized_panda_frame, book_frame)

    return composed_img


def find_best_match(reference_image, other_images):

    best_match_idx = None
    best_match_H = None
    max_inliners = -np.inf
    for i in range(len(other_images)):
        matches, locs1, locs2 = matchPicsORB(reference_image, other_images[i])

        matches = matches[:30]

        H, inliers = computeH_ransac(
            locs1[matches[:, 0]], locs2[matches[:, 1]], 300, 2)
        num_inliers = np.sum(inliers)

        if num_inliers > max_inliners:
            best_match_idx = i
            max_inliners = num_inliers
            best_match_H = H

    return best_match_H, max_inliners, best_match_idx


def warp_image(I1, I2, H):

    I1_h, I1_w = I1.shape[:2]
    I2_h, I2_w = I2.shape[:2]

    H_inv = np.linalg.inv(H)

    # compute the corners of the image1 and image2
    I1_corners = np.float32(
        [[0, 0], [0, I1_h], [I1_w, I1_h], [I1_w, 0]]).reshape(-1, 1, 2)
    I2_corners = np.float32(
        [[0, 0], [0, I2_h], [I2_w, I2_h], [I2_w, 0]])

    I2_corners_hetero = np.concatenate(
        (I2_corners, np.ones((I2_corners.shape[0], 1))), axis=1)

    I2_corners_hetero_projected = np.dot(H_inv, I2_corners_hetero.T).T
    I2_corners_projected = I2_corners_hetero_projected[:,
                                                       :2] / (I2_corners_hetero_projected[:, 2:] + 1e-10)
    I2_corners_projected = I2_corners_projected.reshape(-1, 1, 2)

    # determine the bounding box of the warped image
    all_corner_points = np.concatenate(
        (I1_corners, I2_corners_projected), axis=0)
    [min_x, min_y] = np.int32(all_corner_points.min(axis=0)).reshape(-1)
    [max_x, max_y] = np.int32(all_corner_points.max(axis=0)).reshape(-1)

    # transformation matrix to translate the image
    trans_matrix = np.array(
        [[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])

    # # Warp and combine the images
    warped_img = cv2.warpPerspective(
        I2, trans_matrix @ H_inv, (max_x - min_x, max_y - min_y))
    adjust = [-min_x, -min_y]
    warped_img[adjust[1]:I1_h + adjust[1],
               adjust[0]: I1_w + adjust[0]] = I1

    return warped_img


def build_panorama(selected_pair, reference_image, H):

    warped_image = warp_image(
        selected_pair, reference_image, H)

    # remove extra black regions
    rows_to_remove = []
    cols_to_remove = []
    for row in range(warped_image.shape[0]):
        if np.all(warped_image[row] == 0):
            rows_to_remove.append(row)

    for col in range(warped_image.shape[1]):
        if np.all(warped_image[:, col] == 0):
            cols_to_remove.append(col)

    panorama_img = np.delete(warped_image, rows_to_remove, axis=0)
    panorama_img = np.delete(panorama_img, cols_to_remove, axis=1)

    return panorama_img
