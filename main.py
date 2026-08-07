from src.DataLoader import dataLoader
from src.Center import Center
from src.Covariance import Covariance
from src.EigenSolver import EigenSolver
from src.QRSolver import QRSolver
from src.PCA import PCA
from src.Visualizer import Visualizer
from src.Reconstructor import Reconstructor
from tests.Test_Case import SmallCaseAnalysis


def main():
    print("=== Step 1: Load data ===")
    loader = dataLoader()
    X, t = loader.load()
    loader.get_info()
    loader.show_sample()

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
    visualizer = Visualizer()
    visualizer.plot_cumulative_var(cum_variance)

    print("\n=== Step 7: Dimensionality reduction ===")
    W1 = pca.projection_matrix(eigenvectors, k=k90)
    T1 = pca.transform(B)
    print(f"T shape: {T1.shape}")

    print("\n=== Step 8: 2D visualization ===")
    W2 = pca.projection_matrix(eigenvectors, k=2)
    T2 = pca.transform(B)
    visualizer.plot_2d_scatter(T2, t)
    print(f"T shape: {T2.shape}")

    print("\n=== Step 9: Reconstruction ===")
    reconstructor = Reconstructor()
    k_values = [2, 10, 30]
    results = reconstructor.mse_vs_k(B, eigenvectors, center.mean_vector, X, k_values)
    for k, mse in results:
        print(f"k={k}: MSE={mse:.4f}")

    reconstructions = []
    for k in k_values:
        W_k = eigenvectors[:, :k]
        T_k = B @ W_k
        X_rec = reconstructor.reconstruct(T_k, W_k, center.mean_vector)
        reconstructions.append(X_rec)

    visualizer.plot_mse_vs_k([k for k, _ in results], [mse for _, mse in results])
    visualizer.plot_reconstruction_comparison(X, reconstructions, k_values)

    print("\n=== Step 10: Small sample case (m < n) ===")
    analysis = SmallCaseAnalysis()
    X_small, indices = analysis.random_samples(X, num_samples=50)
    B_small, C_small, eig_small, vec_small = analysis.run_pipeline(X_small)
    analysis.analyze_eigenvalues(eig_small)
    analysis.analyze_space(eig_small)


if __name__ == '__main__':
    main()
