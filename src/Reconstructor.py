class Reconstructor:

    def reconstruct(self, T, W, mean_vector):
        X_reconstructed = T @ W - mean_vector
        return X_reconstructed