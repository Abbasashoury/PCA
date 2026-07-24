import numpy as np


class Center:

    def __init__(self):
        self.mean_vector = None

    def center(self, X):
        self.mean_vector = np.mean(X, axis=0)  # mean vector
        B = X - self.mean_vector  # centeralization
        return B
