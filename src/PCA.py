import numpy as np


class PCA:
    def __init__(self, components):
        self.k = components  # num of components
        self.W = None  # projection matrix
        self.explained_variance_ratio = None
        self.cumulative_variance = None

    def compute_variance(self, eigenvalues):
        total = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues / total
        self.cumulative_variance = np.cumsum(self.explained_variance_ratio)
        return self.explained_variance_ratio, self.cumulative_variance

    def find_k_component(self, target=0.90):
        if self.cumulative_variance is None:
            raise ValueError("Call compute_variance first.")
        k = int(np.searchsorted(self.cumulative_variance, target) + 1)
        print(f"Components need for {target * 100}% variance: {k}")
        return k

    def projection_matrix(self, eigenvectors, k):
        self.k = k
        self.W = eigenvectors[:, :k]

    def transform(self, B):
        if self.W is None:
            raise ValueError("Call projection_matrix first.")
        T = B @ self.W
        return T
