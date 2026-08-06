import numpy as np
from src.Center import Center
from src.Covariance import Covariance
from src.EigenSolver import EigenSolver
class SmallCaseAnalysis:

    def random_samples(self, X, num_samples = 50, seed = 0):

        rng = np.random.default_rng( seed )
        indices = rng.choice(X.shape[0], size=num_samples, replace = False )
        X_small = X[indices]
        return indices , X_small

    def run_pipeline(self, X_small):
        Center = ()
        B_small = Center.center(X_small)

        cov = Covariance()
        C_small = cov.compute(B_small)

        solver = EigenSolver()
        eigenvalues, eigenvectors = solver.solve(C_small)

        return B_small, C_small, eigenvalues, eigenvectors

    def analyze_eigenvalues(self, eigenvalues, tol=1e-8):
        num_zero = int(np.sum((eigenvalues) < tol))
        print(f"Number of (near) zero eigenvalues: {num_zero}")
        print(f"Total eigenvalues: {len(eigenvalues)}")
        return num_zero

    def analyze_space(self, eigenvalues, tol=1e-8):
        rank = int(np.sum((eigenvalues) > tol))
        nullity = len(eigenvalues) - rank
        print(f"rank(C): {rank}")
        print(f"nullity(C): {nullity}")
        return rank, nullity