import numpy as np
import helper as hlp
import matplotlib.pyplot as plt
from Q4triangulation import Q4

# Q5


class Q5(Q4):
    def __init__(self):
        super(Q5, self).__init__()

        """
        Write your own code here 

        Step 1. Complete 'def vis_pts3d' to plot the 3D points computed in Q4.

        Step 2. Load the points from data/some_corresp.npz to compute reprojection error.
                - pts1, pts2
                
        Step 3. Calculate the reprojection error. You should use the triangulate function in Q4.
                - pts3d_corresp
                - reprojection_error = self.compute_reprojerr(self.P1, pts1, self.P2, pts2, pts3d_corresp)

        Step 4. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz. 
                - You should use what you computed in Q4.

        """

        pts1 = np.load('../data/some_corresp.npz')['pts1']
        pts2 = np.load('../data/some_corresp.npz')['pts2']

        pts3d = self.triangulate(self.P1, pts1, self.P2, pts2)

        reproj_err = self.compute_reprojerr(
            self.P1, pts1, self.P2, pts2, pts3d)

        self.reproj_err = reproj_err

        np.savez('../data/extrinsics.npz', R1=self.Ext1[:, :3], R2=self.Ext2[:, :3],
                 t1=self.Ext1[:, 3], t2=self.Ext2[:, 3])

    """
    Q5 Compute Reprojection Error
        [I] P1, camera projection matrix 1 (3x4 matrix)
            pts1, points in image 1 (Nx2 matrix)
            P2, camera projection matrix 2 (3x4 matrix)
            pts2, points in image 2 (Nx2 matrix)
            pts3d, 3D points in space (Nx3 matrix)
        [O] reproj_err, Reprojection Error (float)
    """

    def compute_reprojerr(self, P1, pts1, P2, pts2, pts3d):
        """
        Write your own code here  
        """

        pts3d_homo = np.hstack((pts3d, np.ones((len(pts3d), 1))))
        pts1_pred = P1 @ pts3d_homo.T
        pts1_pred = pts1_pred[:2] / pts1_pred[2]

        reprojerr = np.mean(np.linalg.norm(pts1 - pts1_pred.T, axis=1))

        return reprojerr

    def vis_pts3d(self, pts3d):
        """
        Write your own code here 
        """

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], c='g', marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()


if __name__ == "__main__":

    Q5 = Q5()
    print("Reprojection Error:", Q5.reproj_err)
    Q5.vis_pts3d(Q5.pts3d)
