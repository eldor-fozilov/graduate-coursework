import numpy as np
import helper as hlp
import matplotlib.pyplot as plt
from Q6rectify import Q6
from scipy.signal import convolve2d

class Q7(Q6):
   
    def __init__(self):
        super(Q7, self).__init__()

    
        """
        Write your own code here 

        Step 1. Get disparity with the images rectified at Q6.
            - Set 'min_disp=370' and 'num_disp=60'.
            - dispM = self.get_disparity(I1, I2, window_size, min_disp, num_disp)

        Step 2. Clamp disparity values to the range [min_disp-1, min_disp+num_disp+1].

        Step 3. Complete 'def vis_disparity_image' to visualize the disparity map.

        """

        # DO NOT CHANGE HERE!
        self.disp = dispM

    """
    Q7 Disparity Map
        [I] im1, image 1 (H1xW1 matrix)
            im2, image 2 (H2xW2 matrix)
            win_size, window size value
        [O] dispM, disparity map (H1xW1 matrix)
    """
    def get_disparity(self, I1, I2, win_size, min_disp, num_disp):
        
        """
        Write your own code here 


        

        replace pass by your implementation
        """
        pass


    def vis_disparity_map(self, dispI):

        """
        Write your own code here 


        

        replace pass by your implementation
        """
        pass


if __name__ == "__main__":

    Q7 = Q7()
    Q7.vis_disparity_map(Q7.disp)

