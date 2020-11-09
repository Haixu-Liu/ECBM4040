import time
import numpy as np

def pca_naive(X, K):
    """
    PCA -- naive version

    Inputs:
    - X: (float) A numpy array of shape (N, D) where N is the number of samples,
         D is the number of features
    - K: (int) indicates the number of features you are going to keep after
         dimensionality reduction

    Returns a tuple of:
    - P: (float) A numpy array of shape (K, D), representing the top K
         principal components
    - T: (float) A numpy vector of length K, showing the score of each
         component vector
    """
    V = np.dot(X.T, X)
    
    T, P = np.linalg.eig(V)
    
    idx = np.argsort(T)[::-1]
    idx = idx[:K]

    T = T[idx]
    P = P[:,idx].T
    
    return (P, T)
