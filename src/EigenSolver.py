import numpy as np


class EigenSolver:
    def __init__(self):
        def __init__(self):
            self.eigenvalues = None
            self.eigenvectors = None

    def solveEigen(self, C):
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        order = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[order]
        self.eigenvectors = eigenvectors[:, order]
        return self.eigenvalues, self.eigenvectors
