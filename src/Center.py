import numpy as np


class Center:

    def __init__(self):
        self.mean_vector = None

    def center(self, X):
        self.mean_vector = np.mean(X, axis=0)  # mean vector
        B = X - self.mean_vector  # centeralization
        return B

    def verify_centering(self, B, telorance=1e-8):
        col_means = np.mean(B, axis=0)
        max_mean = np.max(np.abs(col_means))
        is_centered = max_mean < telorance
        print(f"max mean after centering: {max_mean:.2e}")
        print(f"Centered: {is_centered}")
