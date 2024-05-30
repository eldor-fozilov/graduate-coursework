import dis
import numpy as np
from scipy.fftpack import shift
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

        num_disp = 60
        min_disp = 370
        self.window_size = 11
        dispM = self.get_disparity(
            self.I1, self.I2, self.window_size, min_disp, num_disp)

        dispM = np.clip(dispM, min_disp - 1, min_disp + num_disp + 1)
        # Mask out the black regions and unimportant regions to white
        # manually set the threshold to 50
        dispM = np.where(self.I1 <= 50, np.inf, dispM)

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
        """
        H, W = I1.shape
        disp_map = np.zeros((H, W))
        filter = np.ones((win_size, win_size))

        for d in range(min_disp, min_disp + num_disp):

            shifted_I2 = np.zeros_like(I2)
            shifted_I2[:, d:] = I2[:, :-d]  # shift the I2 image to the left

            SSD = (I1 - shifted_I2) ** 2
            dist = convolve2d(SSD, filter, mode='same')

            if d == min_disp:
                min_dist = dist
                disp_map[:] = d
            else:
                check = dist < min_dist
                min_dist = np.where(check, dist, min_dist)
                disp_map = np.where(check, d, disp_map)

        return disp_map

    def vis_disparity_map(self, dispI):
        """
        Write your own code here 
        """
        plt.figure()
        plt.imshow(dispI, cmap='inferno')
        plt.title(f'Disparity Map (window size: {self.window_size})')
        plt.axis('off')
        plt.show()


if __name__ == "__main__":

    Q7 = Q7()
    Q7.vis_disparity_map(Q7.disp)
