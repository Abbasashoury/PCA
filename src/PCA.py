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

    def find_k_component(self, target=0.90):
        if self.cumulative_variance is None:
            raise ValueError("Call compute_variance first.")
        k = int(np.searchsorted(self.cumulative_variance, target) + 1)
        print(f"Components need for {target * 100}% variance: {k}")
        return k

    def projection_matrix(self, eigenvectors, k):
        self.k = k
        self.W = eigenvectors[:, :k]
