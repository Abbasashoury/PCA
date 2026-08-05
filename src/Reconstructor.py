import numpy as np

class Reconstructor:

    def reconstruct(self, T, W, mean_vector):
        
        X_reconstructed = T @ W.T + mean_vector
        return X_reconstructed

    def compute_mse(self, X_original, X_reconstructed):

        mse = np.mean(X_original - X_reconstructed)
        return mse

    def mse_vs_k(self, B, eigenvectors, mean_vector, X_original, k_values):
        results = []
        for k in k_values:
            W_k = eigenvectors[:, :k]
            T_k = B @ W_k
            X_rec = self.reconstruct(T_k, W_k, mean_vector)
            mse = self.compute_mse(X_original, X_rec)
            results.append((mse,k))
        return results