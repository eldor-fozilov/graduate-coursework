from cycler import K
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection as poly
from mpl_toolkits.mplot3d.art3d import Poly3DCollection as poly3d
from scipy.datasets import face
from Q10estimate_params import Q10

# Q11


class Q11(Q10):
    def __init__(self):
        super(Q11, self).__init__()

        """
        Write your own code here 

        Step 1. Load the mesh file '../data/pnp.npz'.
                - {X, x, image, cad, vertices, faces}

        Step 2. Estimate the camera pose and parameters using your code from Q9 and Q10. 
                - You can use self.estimate_pose() or self.estimate_params().

        Step 3. Use your estimated camera matrix P to project the given 3D points X onto the image.

        Step 4. Plot the given 2D points x and the projected 3D points on screen.
        
        Step 5. Draw the CAD model rotated by your estimated rotation R on screen.

        Step 6. Project the CAD’s all vertices onto the image, and draw the projected CAD model overlapping with the 2D image. 

        """

        # Load the mesh file
        data = np.load('../data/pnp.npz', allow_pickle=True)
        X, x = data['X'], data['x']  # X: 3D points, x: 2D points
        image, CAD = data['image'], data['cad']
        # vertices: 3D vertices, faces: 3D faces [make the vertex index start from 0]
        vertices, faces = CAD['vertices'][0][0], CAD['faces'][0][0] - 1

        # Estimate the camera pose and parameters
        P = self.estimate_pose(x, X)
        K, R, t = self.estimate_params(P)

        # Project the given 3D points X onto the image
        # convert to homogeneous coordinates
        X_homo = np.hstack((X, np.ones((X.shape[0], 1))))
        x_proj_homo = P @ X_homo.T
        x_proj = (x_proj_homo[:2] / x_proj_homo[2]).T

        # Plot the 2D points and the projected 3D points
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.scatter(x[:, 0], x[:, 1], s=100,
                   edgecolors='g', facecolors='none')
        ax.scatter(x_proj[:, 0], x_proj[:, 1], s=10, c='black')

        # Rotate the CAD model
        CAD_rotated = R @ vertices.T
        X_cor, Y_cor, Z_cor = CAD_rotated

        # Plot the rotated CAD model
        fig_CAD = plt.figure()
        ax_CAD = fig_CAD.add_subplot(111, projection='3d')

        poly_collect = poly3d(
            CAD_rotated.T[faces], edgecolors='blue', alpha=0)

        ax_CAD.add_collection3d(poly_collect)
        ax_CAD.set_xlim(min(X_cor), max(X_cor))
        ax_CAD.set_ylim(min(Y_cor), max(Y_cor))
        ax_CAD.set_zlim(min(Z_cor), max(Z_cor))

        # Project CAD vertices onto the imageS
        CAD_homo = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        CAD_proj_homo = P @ CAD_homo.T
        CAD_proj = CAD_proj_homo[:2] / CAD_proj_homo[2]

        # Plot the projected CAD model on the image
        fig, ax_comb = plt.subplots()
        ax_comb.imshow(image)
        for face_idx in range(len(faces)):
            ax_comb.fill(CAD_proj[0, faces[face_idx]], CAD_proj[1,
                         faces[face_idx]], facecolor='red', alpha=0.25)
        plt.show()


if __name__ == "__main__":

    Q11 = Q11()
