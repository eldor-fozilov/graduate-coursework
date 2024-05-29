import numpy as np
import matplotlib.pyplot as plt
from Q7disparity import Q7

# Q8


class Q8(Q7):

    def __init__(self):
        super(Q8, self).__init__()

        """
        Write your own code here 

        Step 1. Load the rectify parameters from '../data/rectify.npz'.
            - M1, M2, K1p, K2p, R1p, R2p, t1p, t2p

        Step 2. Get depth map with the disparity map and the camera parameters.
            - self.get_depth()

        Step 3. Complete 'def vis_depth_map' to visualize the depth map.

        """

        rectify_params = np.load('../data/rectify.npz')
        K1p = rectify_params['K1p']
        K2p = rectify_params['K2p']
        R1p = rectify_params['R1p']
        R2p = rectify_params['R2p']
        t1p = rectify_params['t1p']
        t2p = rectify_params['t2p']

        depthM = self.get_depth(self.disp, K1p, K2p, R1p, R2p, t1p, t2p)
        # Mask out the black regions and unimportant regions to white
        # manually set the threshold to 50
        depthM = np.where(self.I1 <= 50, np.inf, depthM)

        # DO NOT CHANGE HERE!
        self.depth = depthM

    """
    Q8 Depth Map
        [I] dispM, disparity map (H1xW1 matrix)
            K1 K2, camera matrices (3x3 matrix)
            R1 R2, rotation matrices (3x3 matrix)
            t1 t2, translation vectors (3x1 matrix)
        [O] depthM, depth map (H1xW1 matrix)
    """

    def get_depth(self, dispM, K1p, K2p, R1p, R2p, t1p, t2p):
        """
        Write your own code here 
        """
        c1 = np.matmul(np.linalg.inv(K1p @ R1p), K1p @ t1p)
        c2 = np.matmul(np.linalg.inv(K2p @ R2p), K2p @ t2p)
        b = np.linalg.norm(c1 - c2, axis=0).item()
        f = K1p[1, 1]

        mask = dispM == 0
        # handle the case where disparity is 0: depth will be 0
        dispM[mask] = np.inf
        depthM = b * f / dispM

        return depthM

    def vis_depth_map(self, depthM):
        """
        Write your own code here 
        """
        plt.figure()
        plt.imshow(depthM, cmap='inferno')
        plt.title(f'Depth Map (window size: {self.window_size})')
        plt.axis('off')
        plt.show()


if __name__ == "__main__":

    Q8 = Q8()
    Q8.vis_depth_map(Q8.depth)
