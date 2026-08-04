import matplotlib.pyplot as plt

class Visualizer:

    def plot_sample_image(self, X, y, index=0):
        image = X[index].reshape(8, 8)
        plt.figure(figsize=(3, 3))
        plt.imshow(image, cmap="gray")
        plt.title(f"Label: {X[index]}")
        plt.axis("off")
        plt.show()