import matplotlib.pyplot as plt

class Visualizer:

    def plot_sample_image(self, X, y, index=0):
        image = X[index].reshape(8, 8)
        plt.figure(figsize=(3, 3))
        plt.imshow(image, cmap="gray")
        plt.title(f"Label: {X[index]}")
        plt.axis("off")
        plt.show()

    def plot_cumulative_var(self, cumulative_variance, target=0.90):
        plt.figure(figsize=(6, 4))
        plt.plot(range(1 , len(cumulative_variance) + 1) , cumulative_variance , marker="o", markersize=3)
        plt.axhline(y=target , color="r", linestyle="--", label=f"{target * 100:.0f}% variance")
        plt.xlabel("Number of components")
        plt.ylabel("Cumulative explained variance")
        plt.title("Cumulative Explained Variance")
        plt.legend()
        plt.grid(True)
        plt.savefig("cumulative_variance.png")
        plt.show()

    def plot_2d_scatter(self, T2, y):
        plt.figure(figsize=(7, 6))
        scatter = plt.scatter(T2[:, 0], T2[:, 1], c=y, cmap="tab10", s=15)
        plt.colorbar(scatter , label="Digit label")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("2D PCA Projection")
        plt.grid(True)
        plt.savefig("scatter_2d.png")
        plt.show()

    def plot_mse_vs_k(self, k_values, mse_values):
        plt.figure(figsize=(6, 4))
        plt.plot(k_values, mse_values, marker="o")
        plt.xlabel("Number of components (k)")
        plt.ylabel("MSE")
        plt.title("Reconstruction Error vs k")
        plt.grid(True)
        plt.show()

    def plot_reconstruction_comparison(self, X_orig, reconstructions, k_values, index = 0):
        n_plots = len(k_values) + 1
        plt.figure(figsize=(3 * n_plots, 3))

        plt.subplot(1, n_plots, 1)
        plt.imshow(X_orig[index].reshape(8, 8), cmap="gray")
        plt.title("Original")
        plt.axis("off")
        for i, (k, X_rec) in enumerate(zip(k_values ,reconstructions )):
            plt.subplot(1, n_plots, i + 2)
            plt.imshow(X_rec[index].reshape(8, 8), cmap="gray")
            plt.title(f"k={k}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()