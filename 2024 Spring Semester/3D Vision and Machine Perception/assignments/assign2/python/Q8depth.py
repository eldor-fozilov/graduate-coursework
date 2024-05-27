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


        

        replace pass by your implementation
        """
        pass

    def vis_depth_map(self, depthM):

        """
        Write your own code here 


        

        replace pass by your implementation
        """
        pass

if __name__ == "__main__":

    Q8 = Q8()
    Q8.vis_depth_map(Q8.depth)