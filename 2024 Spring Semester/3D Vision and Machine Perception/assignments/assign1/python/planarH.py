from matplotlib import scale
import numpy as np
import cv2


def bilinear_interpolation(image, x, y):

    x0 = int(x)
    y0 = int(y)
    x1 = x0 + 1
    y1 = y0 + 1

    # Ensure the coordinates are within the image bounds
    x0 = max(0, min(x0, image.shape[1] - 1))
    y0 = max(0, min(y0, image.shape[0] - 1))
    x1 = max(0, min(x1, image.shape[1] - 1))
    y1 = max(0, min(y1, image.shape[0] - 1))

    # Compute the fractional parts
    dx = x - x0
    dy = y - y0

    # Perform bilinear interpolation
    interpolated_value = (1 - dx) * (1 - dy) * image[y0, x0] + \
        dx * (1 - dy) * image[y0, x1] + \
        (1 - dx) * dy * image[y1, x0] + \
        dx * dy * image[y1, x1]

    return interpolated_value


def computeH(x1, x2):
    # Q8
    # Compute the homography between two sets of points

    A = []
    num_points = x1.shape[0]

    for i in range(num_points):
        x_, y_ = x1[i]
        x, y = x2[i]
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
            locs = locs / (locs[2] + 1e-10)
            x_loc, y_loc = locs[0], locs[1]

            # check the bounds
            if 0 <= x_loc < input_width and 0 <= y_loc < input_height:
                # color the points
                img_warped[y, x] = bilinear_interpolation(img, x_loc, y_loc)
                mask[y, x] = 1
            else:
                mask[y, x] = 0

    return img_warped, mask


def computeH_norm(x1, x2):
    # Q9
    # Compute the centroid of the points
    x1_mean = np.mean(x1, axis=0)
    x2_mean = np.mean(x2, axis=0)
    # Shift the origin of the points to the centroid
    x1_norm = x1 - x1_mean
    x2_norm = x2 - x2_mean
    # Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    x1_max = np.max(np.linalg.norm(x1_norm, axis=1))
    x2_max = np.max(np.linalg.norm(x2_norm, axis=1))

    x1_scale = np.sqrt(2) / x1_max  # set the largest distance to sqrt(2)
    x2_scale = np.sqrt(2) / x2_max  # set the largest distance to sqrt(2)

    x1_norm = x1_norm * x1_scale
    x2_norm = x2_norm * x2_scale

    # Similarity transform 1
    T_1 = np.array([[x1_scale, 0, -x1_scale * x1_mean[0]],
                   [0, x1_scale, -x1_scale * x1_mean[1]],
                   [0, 0, 1]])
    # Similarity transform 2
    T_2 = np.array([[x2_scale, 0, -x2_scale * x2_mean[0]],
                   [0, x2_scale, -x2_scale * x2_mean[1]],
                   [0, 0, 1]])
    # Compute homography
    H2to1 = computeH(x1_norm, x2_norm)

    # Denormalization
    H2to1 = np.dot(np.linalg.inv(T_1) @ H2to1, T_2)

    return H2to1


def computeH_ransac(locs1, locs2, max_iter, threshold):
    # Q10
    # Compute the best fitting homography given a list of matching points

    # Extend the locs1 and locs2 to homogeneous coordinates by adding 1
    locs1_h = np.hstack((locs1, np.ones((len(locs1), 1))))
    locs2_h = np.hstack((locs2, np.ones((len(locs2), 1))))

    # Initialize the variables
    bestH2to1 = None
    max_num_inliers = -np.inf
    max_inliers = None

    for iter in range(max_iter):

        rand_indices = np.random.choice(len(locs1), 4, replace=False)

        locs1_rand = locs1[rand_indices]
        locs2_rand = locs2[rand_indices]

        H2to1 = computeH_norm(locs1_rand, locs2_rand)

        homo_locs = H2to1 @ locs2_h.T  # shape (3, N)
        hetero_locs = homo_locs[:2] / (homo_locs[2] + 1e-10)

        hetero_locs = hetero_locs.T  # back to shape (N, 2)

        inliers = np.linalg.norm(
            locs1 - hetero_locs, axis=1) <= threshold

        num_inliers = np.sum(inliers)

        if num_inliers > max_num_inliers:
            bestH2to1 = H2to1
            max_num_inliers = num_inliers
            max_inliers = inliers

    inliers = max_inliers

    return bestH2to1, inliers


def compositeH(H2to1, template, img):

    # Create a composite image after warping the template image on top
    # of the image using the homography

    warped_template, mask = warpPerspective(
        template, H2to1, (img.shape[0], img.shape[1]))

    warped_template[mask == 0] = img[mask == 0]

    return warped_template
