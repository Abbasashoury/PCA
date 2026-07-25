import numpy as np


class PCA:
    def __init__(self, components):
        self.k = components  # num of components
        self.W = None  # projection matrix
        self.relative_variance = None
        self.cumulative_variance = None

    def compute_variance(self, eigenvalues):
        total = np.sum(eigenvalues)
        self.relative_variance = eigenvalues / total
        self.cumulative_variance = np.cumsum(self.relative_variance)
        return self.relative_variance, self.cumulative_variance
