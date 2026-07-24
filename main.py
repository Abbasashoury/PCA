from src.Data_loader import DataLoader
from src.Center import Center
from src.Covariance import Covariance
from src.EigenSolver import EigenSolver


def main():
    print("=== Step 1: Load data ===")
    loader = DataLoader()
    X, t = loader.load()
    loader.get_info()

    print("\n=== Step 2: Center data ===")
    center = Center()
    B = center.center(X)
    center.verify_centering(B)

    print("\n=== Step 3:Covariance matrix ====")
    cov = Covariance()
    C = cov.compute(B)
    cov.check_symmetry(C)
    cov.get_shape(C)

    print("\n=== Step 5: Eigen decomposition ===")
    solver = EigenSolver()
    eigenvalues, eigenvectors = solver.solveEigen(C)
    solver.verify_orthogonality()


if __name__ == '__main__':
    main()
