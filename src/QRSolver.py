import numpy as np

class QRSolver:
    def qr_iteration(self, A, num_iterations=1):
        A_k = A.copy()
        for _ in range(num_iterations):
            R,Q = np.linalg.qr(A_k)
            A_k = R @ Q
        return A_k

    def test_similarity(self, A_original, A_new):

        eig_orig = np.sort(np.linalg.eigvalsh(A_original))[::-1]
        eig_new = np.sort(np.linalg.eigvalsh(A_new))
        max_diff = np.max((eig_orig - eig_new))
        print(f"Original eigenvalues:{eig_orig} ")
        print(f" New eigenvalues:    {eig_new} ")
        print(f" Max difference:{max_diff:.2e} ")
        return max_diff

    def compute_rank_R(self, B, tol=1e-10):
        Q , R = np.linalg.qr(B, mode="reduced")
        diag_R = (np.diag(R))
        rank = int(np.sum(diag_R >= tol))
        print(f" Diagonal of R:{diag_R} ")
        print(f"Estimated rank: {rank} ")
