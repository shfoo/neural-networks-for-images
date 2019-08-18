import numpy as np

class PCA(object):
    def __init__(self):
        self.total_mean = None
        self.principal_components = None

    def train(self, X):
        """
        Input:
        - X: Array of shape (N, D)
        """
        total_mean = np.mean(X, axis=0)
        X_ = (X - total_mean).T

        # X is (D, N) when used to compute SVD
        U, D, V_T = np.linalg.svd(X_, full_matrices=False)

        self.total_mean = total_mean
        self.principal_components = U.T

    def test(self, X, k=200):
        """
        Inputs:
        - X: Array on which to perform PCA, of shape (N, D)
        - k: Number of principal components to use

        Output:
        - Data points in X projected onto the principal components,
          contained in array of shape (N, k)
        """
        return np.dot(X, (self.principal_components.T)[:, 0:k])
