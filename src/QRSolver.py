import numpy as np

class QRSolver:
    def qr_iteration(self, A, num_iterations=1):
        A_k = A.copy()
        for _ in range(num_iterations - 1):
            R,Q = np.linalg.qr(A_k)
            A_k = R @ Q
        return A_k
