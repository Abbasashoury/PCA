class Covariance:
    def __init__(self):
        pass

    def computeCovariance(self, B):
        m = B.shape[0]  # num of sample
        C = (1.0 / (m - 1)) * (B.T @ B)
        return C
