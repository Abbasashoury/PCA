import numpy as np


class EigenSolver:
    def __init__(self):
        self.eigenvalues = None
        self.eigenvectors = None

    def solveEigen(self, C):
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        order = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[order]
        self.eigenvectors = eigenvectors[:, order]
        return self.eigenvalues, self.eigenvectors

    def verify_orthogonality(self, tolerance=1e-8):
        W = self.eigenvectors
        product = W.T @ W
        identity = np.eye(W.shape[1])
        max_diff = np.max(np.abs(product - identity))
        is_orthogonal = max_diff < tolerance
        print(f"Max |W^T.W-I|: {max_diff}")
        print(f"Orthogonal: {is_orthogonal:.2e}")
