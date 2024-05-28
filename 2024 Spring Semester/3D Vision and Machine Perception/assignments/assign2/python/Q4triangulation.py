import numpy as np
import helper as hlp
from Q3essential_matrix import Q3

# Q4


class Q4(Q3):
    def __init__(self):
        super(Q4, self).__init__()

        """
        Write your own code here 

        Step 1. Compute the camera projection matrices P1 using self.K1
            - P1

        Step 2. Use hlp.camera2 to get 4 camera projection matrices P2
            - P2s

        for loop range of 4:
        
            Step 3. Run triangulate using the projection matrices
                - pts3d = self.triangulate(P1, self.pts1, P2, self.pts2)

            Step 4. Figure out the correct P2
                - P2

        Step 5. Compose the extrinsic matrices and 3D points using the correct P2.
            - Ext1, Ext2, pts3d
        """

        # Ext1 = [I | 0] (3x4 matrix)
        Ext1 = np.concatenate([np.identity(3), np.zeros((3, 1))], axis=1)
        P1 = np.concatenate([self.K1, np.zeros((3, 1))],
                            axis=1)  # P1 = [K | 0] (3x4 matrix)

        Ext2_cands = hlp.camera2(self.E)
        num_cands = len(Ext2_cands)

        valid_3d_pts_num_for_cands = {}

        for cand_idx in range(num_cands):

            P2_cand = self.K2 @ Ext2_cands[cand_idx]
            pts3d = self.triangulate(P1, self.pts1, P2_cand, self.pts2)

            # Count the number of valid 3D points
            valid_3d_pts_num_for_cands[f'P2 Candidate {cand_idx + 1}'] = np.sum(
                pts3d[:, 2] > 0)

        print(valid_3d_pts_num_for_cands)
        max_idx = np.argmax(list(valid_3d_pts_num_for_cands.values()))
        print("Chosen P2 Candidate:", max_idx + 1)

        Ext2 = Ext2_cands[max_idx]
        P2 = self.K2 @ Ext2

        # DO NOT CHANGE HERE!
        self.Ext1, self.Ext2 = Ext1, Ext2
        self.P1, self.P2 = P1, P2
        self.pts3d = pts3d

    """
    Q4 Triangulation
        [I] P1, camera projection matrix 1 (3x4 matrix)
            pts1, points in image 1 (Nx2 matrix)
            P2, camera projection matrix 2 (3x4 matrix)
            pts2, points in image 2 (Nx2 matrix)
        [O] pts3d, 3D points in space (Nx3 matrix)
    """

    def triangulate(self, P1, pts1, P2, pts2):
        """
        Write your own code here 
        """
        pts3d = np.zeros(((len(pts1), 3)))
        for i in range(len(pts1)):

            A = np.array([
                pts1[i, 1] * P1[2, :] - P1[1, :],
                P1[0, :] - pts1[i, 0] * P1[2, :],
                pts2[i, 1] * P2[2, :] - P2[1, :],
                P2[0, :] - pts2[i, 0] * P2[2, :]])

            U, S, V_T = np.linalg.svd(A)
            pts3d_homo = V_T[-1, :]
            pts3d_hetero = pts3d_homo[:3] / pts3d_homo[3]
            pts3d[i] = pts3d_hetero

        return pts3d


if __name__ == "__main__":

    Q4 = Q4()
    print("Ext2=", Q4.Ext2)
