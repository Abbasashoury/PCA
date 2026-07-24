import numpy as np


class Covariance:
    def __init__(self):
        pass

    def computeCovariance(self, B):
        m = B.shape[0]  # num of sample
        C = (1.0 / (m - 1)) * (B.T @ B)
        return C

    def check_symmetry(self, C, telorance):
        is_symmetric = np.allclose(C, C.T, atol=telorance)
        print(f"Symmetric: {is_symmetric}")