import numpy as np
from sklearn.datasets import load_digits

class dataLoader:
    """
    step 1
    data loading...
    """
    def __init__(self):
        self.X = None # main matrix
        self.t = None # tag: number
        self.s = None # num of sample
        self.f = None # num of feature

    def load(self):
        digits = load_digits()
        self.X = digits.data # matrix: (1800,64)
        self.t = digits.target # tag: (0,9)
        self.s, self.f = self.X.shape
        return self.X, self.t

    def get_info(self):
        print("m (samples):", self.s)
        print("n (features):", self.f)
        print("X Shape:", self.X.shape)
        print("y Shape:", self.t.shape)
        print("unique labels", np.unique(self.t))
