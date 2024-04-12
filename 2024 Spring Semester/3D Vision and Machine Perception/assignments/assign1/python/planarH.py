import numpy as np
import cv2


def computeH(x1, x2):
    # Q8
    # Compute the homography between two sets of points

    A = []
    num_points = x1.shape[0]

    for i in range(num_points):
        x, y = x1[i]
        x_, y_ = x2[i]
        A.extend([[-x, -y, -1, 0, 0, 0, x*x_, y*x_, x_],
                  [0, 0, 0, -x, -y, -1, x*y_, y*y_, y_]])

    A = np.array(A)
    U, S, V = np.linalg.svd(A)
    H2to1 = V[-1].reshape(3, 3)

    return H2to1


def warpPerspective(img, H, size_warped):
    # Q8
    # warp image using pre-computed homography matrix

    input_height, input_width = img.shape[:2]
    output_height, output_width = size_warped

    # Create arrays to store coordinates of output image and mask
    img_warped = np.zeros(
        (output_height, output_width, img.shape[2]), dtype=img.dtype)
    mask = np.zeros((output_height, output_width))

    # compute inverse of homography matrix
    H_inv = np.linalg.inv(H)

    for x in range(output_width):
        for y in range(output_height):
            # backward warping
            locs = np.dot(H_inv, np.array([x, y, 1]))
            locs = locs / locs[2]  # normalization
            x_loc, y_loc = int(locs[0]), int(locs[1])

            # check the bounds
            if 0 <= x_loc < input_width and 0 <= y_loc < input_height:
                # color the points
                img_warped[y, x] = img[y_loc, x_loc]
                mask[y, x] = 1
            else:
                mask[y, x] = 0

    return img_warped, mask


def computeH_norm(x1, x2):
    # Q9
    # Compute the centroid of the points

    # Shift the origin of the points to the centroid

    # Normalize the points so that the largest distance from the origin is equal to sqrt(2)

    # Similarity transform 1

    # Similarity transform 2

    # Compute homography

    # Denormalization

    return H2to1


def computeH_ransac(locs1, locs2, max_iter, threshold):
    # Q10
    # Compute the best fitting homography given a list of matching points

    return bestH2to1, inliers


def compositeH(H2to1, template, img):

    # Create a composite image after warping the template image on top
    # of the image using the homography

    # Note that the homography we compute is from the image to the template;
    # x_template = H2to1*x_photo
    # For warping the template to the image, we need to invert it.

    # Create mask of same size as template

    # Warp mask by appropriate homography

    # Warp template by appropriate homography

    # Use mask to combine the warped template and the image

    return composite_img
