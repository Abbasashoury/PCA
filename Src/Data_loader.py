from sklearn.datasets import load_digits

class DataLoader:
    """
    step 1
    data loading...
    """
    def __init__(self):
        self.X = None # main matrix
        self.y = None # tag: number
        self.m = None # num of sample
        self.n = None # num of feature

    def load(self):
        digits = load_digits()
        self.X = digits.data # matrix: (1800,64)
        self.y = digits.target # tag: (0,9)
        self.m, self.n = self.X.shape
        return self.X, self.y
