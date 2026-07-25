from src.DataLoader import dataLoader
from src.Center import Center
from src.Covariance import Covariance
from src.EigenSolver import EigenSolver
from src.PCA import PCA


def main():
    print("=== Step 1: Load data ===")
    loader = dataLoader()
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

    print("\n=== Step 6 :Explained variance == =")
    pca = PCA(components=10)
    rel_variance, cum_variance = pca.compute_variance(eigenvalues)
    k90 = pca.find_k_component(target=0.90)


if __name__ == '__main__':
    main()
