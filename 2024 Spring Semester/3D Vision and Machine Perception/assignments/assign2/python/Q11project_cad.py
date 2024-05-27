import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection as poly
from mpl_toolkits.mplot3d.art3d import Poly3DCollection as poly3d
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

if __name__ == "__main__":

    Q11 = Q11()