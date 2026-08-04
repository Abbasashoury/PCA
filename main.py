from src.DataLoader import dataLoader
from src.Center import Center
from src.Covariance import Covariance
from src.EigenSolver import EigenSolver
from src.QRSolver import QRSolver
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

    print("\n=== Step 3: Covariance matrix ===")
    cov = Covariance()
    C = cov.computeCovariance(B)
    cov.check_symmetry(C)
    cov.get_shape(C)

    print("\n=== Step 4: QR exercise ===")
    qr = QRSolver()
    qr.run_small_example()
    qr.compute_rank_R(B)


    print("\n=== Step 5: Eigen decomposition ===")
    solver = EigenSolver()
    eigenvalues, eigenvectors = solver.solveEigen(C)
    solver.verify_orthogonality()

    print("\n=== Step 6: Explained variance ===")
    pca = PCA(components=10)
    rel_variance, cum_variance = pca.compute_variance(eigenvalues)
    k90 = pca.find_k_component(target=0.90)

    print("\n=== Step 7: Dimensionality reduction ===")
    W1 = pca.projection_matrix(eigenvectors, k=k90)
    T1 = pca.transform(B)
    print(f"T shape: {T1.shape}")

    print("\n=== Step 8: 2D visualization ===")
    W2 = pca.projection_matrix(eigenvectors, k=2)
    T2 = pca.transform(B)
    print(f"T shape: {T2.shape}")


if __name__ == '__main__':
    main()
