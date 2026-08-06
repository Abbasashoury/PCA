import numpy as np

class SmallCaseAnalysis:

    def random_samples(self, X, num_samples = 50, seed = 0):

        rng = np.random.default_rng( seed )
        indices = rng.choice(X.shape[0], size=num_samples, replace = True )
        X_small = X[indices]
        return indices , X_small
