class PCA:
    def __init__(self, components):
        self.k = components  # num of components
        self.W = None  # projection matrix
        self.relative_variance = None
        self.cumulative_variance = None
