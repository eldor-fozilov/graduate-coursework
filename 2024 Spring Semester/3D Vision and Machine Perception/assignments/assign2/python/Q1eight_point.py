import numpy as np
import cv2
import helper as hlp


# Q1

class Q1:
    def __init__(self):

        """
        Write your own code here 

        Step 1. Load the two temple images and the points from data/some_corresp.npz
            - im1, im2, pts

        Step 2. Run eight_point to compute F
            - F = self.eight_point(pts1, pts2, M)

        """

        im1 = cv2.imread('../data/im1.png')
        im2 = cv2.imread('../data/im2.png')
        pts = np.load('../data/some_corresp.npz')
        pts1 = pts['pts1']
        pts2 = pts['pts2']
        M = max(im1.shape[0], im1.shape[1])
        F = self.eight_point(pts1, pts2, M)
    

        # DO NOT CHANGE HERE!
        self.im1 = im1
        self.im2 = im2
        self.F = F


    """
    Q1 Eight Point Algorithm
        [I] pts1, points in image 1 (Nx2 matrix)
            pts2, points in image 2 (Nx2 matrix)
            M, scalar value computed as max(H1,W1)
        [O] F, the fundamental matrix (3x3 matrix)
    """
    def eight_point(self, pts1, pts2, M):

        """Write your code here"""

        # Normalize the points
        pts1 = pts1 / M # for simplicity, we don't convert to homogeneous coordinates
        pts2 = pts2 / M # for simplicity, we don't convert to homogeneous coordinates

        # Construct the A matrix
        A = np.zeros((pts1.shape[0], 9)) # N x 9 matrix
        for i in range(pts1.shape[0]):
            A[i] = [pts1[i, 0] * pts2[i, 0], pts1[i, 0] * pts2[i, 1],
                    pts1[i, 0], pts1[i, 1] * pts2[i, 0], pts1[i, 1] * pts2[i, 1],
                    pts1[i, 1], pts2[i, 0], pts2[i, 1], 1]
            
        # Compute the SVD of A
        U, S, V = np.linalg.svd(A)
        F = V[-1].reshape(3, 3)
        # Compute the SVD of F
        u, s, v = np.linalg.svd(F)
        # Set the rank of F to 2
        s[2] = 0
        # Reconstruct F
        F = u @ np.dot(np.diag(s), v)
        # Refine F
        F = hlp.refineF(F, pts1, pts2)
        # Unnormalize the fundamental matrix
        T = np.diag([1 / M, 1 / M , 1]) # transformation matrix
        F = T.T @ np.dot(F, T)
        
        print("-"*50)
        print("Recovered Fundamental Matrix")
        print(F)
        print("-"*50)
        
        return F


if __name__ == "__main__":

    Q1 = Q1()
    hlp.displayEpipolarF(Q1.im1, Q1.im2, Q1.F)


