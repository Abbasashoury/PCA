import numpy as np
from src.Center import Center
from src.Covariance import Covariance
from src.EigenSolver import EigenSolver
class SmallCaseAnalysis:

    def random_samples(self, X, num_samples = 50, seed = 0):

        rng = np.random.default_rng( seed )
        indices = rng.choice(X.shape[0], size=num_samples, replace = True )
        X_small = X[indices]
        return indices , X_small

    def run_pipeline(self, X_small):
        Center = ()
        B_small = Center.center(X_small)

        cov = Covariance()
        C_small = cov.compute(X_small)

        solver = EigenSolver()
        eigenvalues, eigenvectors = solver.solve(C_small)

        return B_small, C_small, eigenvalues, eigenvectors