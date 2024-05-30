import numpy as np
import numpy.linalg as la
from Q9estimate_pose import Q9

# Q10

# set the random seed to have the same results
np.random.seed(0)


class Q10(Q9):
    # DO NOT CHANGE HERE!
    def __init__(self):
        super(Q10, self).__init__()

        # Test parameter estimation with clean points
        Pc = self.Pc
        Kc, Rc, tc = self.estimate_params(Pc)

        # Test parameter estimation with noisy points
        Pn = self.Pn
        Kn, Rn, tn = self.estimate_params(Pn)

        self.Kc, self.Rc, self.tc = Kc, Rc, tc
        self.Kn, self.Rn, self.tn = Kn, Rn, tn

    """
    Q10 Camera Parameter Estimation
        [I] P, camera matrix (3x4 matrix)
        [O] K, camera intrinsics (3x3 matrix)
            R, camera extrinsics rotation (3x3 matrix)
            t, camera extrinsics translation (3x1 matrix)
    """

    def estimate_params(self, P):
        """
        Write your own code here
        """
        U, S, V_T = np.linalg.svd(P)
        c = V_T[-1]
        c = c[:-1] / c[-1]
        # Remove the last column - 3x4 -> 3x3
        M = P[:, :-1]

        QR_decomp = la.qr(np.linalg.inv(M))
        Q = QR_decomp.Q
        R = QR_decomp.R

        K = np.linalg.inv(R)  # 3x3
        R = np.linalg.inv(Q)  # 3x3

        # Adjust signs of K and R to ensure positive diagonal of K
        D = np.diag(np.sign(np.diag(K)))
        K = K @ D
        # Normalize the matrix so that the last element to 1
        K = K / K[-1, -1]
        R = D @ R
        t = -np.matmul(R, c)

        return K, R, t


if __name__ == "__main__":

    Q10 = Q10()

    print('Intrinsic Error with clean 2D points:', la.norm(
        (Q10.Kc/Q10.Kc[-1, -1])-(Q10.K/Q10.K[-1, -1])))
    print('Rotation Error with clean 2D points:', la.norm(Q10.R-Q10.Rc))
    print('Translation Error with clean 2D points:', la.norm(Q10.t-Q10.tc))

    print('Intrinsic Error with noisy 2D points:', la.norm(
        (Q10.Kn/Q10.Kn[-1, -1])-(Q10.K/Q10.K[-1, -1])))
    print('Rotation Error with noisy 2D points:', la.norm(Q10.R-Q10.Rn))
    print('Translation Error with noisy 2D points:', la.norm(Q10.t-Q10.tn))
