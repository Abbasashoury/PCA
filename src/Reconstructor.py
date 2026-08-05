import numpy as np

class Reconstructor:

    def reconstruct(self, T, W, mean_vector):
        
        X_reconstructed = T @ W.T - mean_vector
        return X_reconstructed

    def compute_mse(self, X_original, X_reconstructed):

        mse = np.mean(X_original - X_reconstructed)
        return mse