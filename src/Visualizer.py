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
        plt.plot(range(0 , len(cumulative_variance) + 1) , cumulative_variance , marker="o", markersize=3)
        plt.axhline(y=target * 100, color="r", linestyle="--", label=f"{target * 100:.0f}% variance")
        plt.xlabel("Number of components")
        plt.ylabel("Cumulative explained variance")
        plt.title("Cumulative Explained Variance")
        plt.legend()
        plt.grid(True)
        plt.show()