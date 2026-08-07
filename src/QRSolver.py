import numpy as np


class QRSolver:
    def qr_iteration(self, A, num_iterations=1):
        A_k = A.copy()
        for _ in range(num_iterations):
            Q, R = np.linalg.qr(A_k)
            A_k = R @ Q
        return A_k

    def test_similarity(self, A_original, A_new):
        eig_orig = np.sort(np.linalg.eigvalsh(A_original))[::-1]
        eig_new = np.sort(np.linalg.eigvalsh(A_new))[::-1]
        max_diff = np.max(np.abs(eig_orig - eig_new))
        print(f"Original eigenvalues:{eig_orig} ")
        print(f" New eigenvalues:    {eig_new} ")
        print(f" Max difference:{max_diff:.2e} ")
        return max_diff

    def compute_rank_R(self, B, tolerance=1e-10):
        Q, R = np.linalg.qr(B, mode="reduced")
        diag_R = np.abs(np.diag(R))
        rank = int(np.sum(diag_R > tolerance))
        print(f" Diagonal of R:\n{diag_R} ")
        print(f"Estimated rank: {rank} ")
        return rank

    def run_small_example(self, size=4, num_iterations=50, seed=0):
        rng = np.random.default_rng(seed)
        M = rng.standard_normal((size, size))
        A = (M + M.T) / 2
        A_new = self.qr_iteration(A, num_iterations=num_iterations)
        self.test_similarity(A, A_new)
        print("\nMatrix after QR iterations: ")
        print(np.round(A_new, 4))
        return A, A_new
