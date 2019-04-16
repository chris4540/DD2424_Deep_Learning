"""
Preprocessing module for assignment 2 Exercise 1
"""
import numpy as np

def normalize_data(X_mat):
    mean_X = np.mean(X_mat, axis=1)[:, np.newaxis]
    std_X = np.std(X_mat,  axis=1)[:, np.newaxis]
    normalized_X_mat = (X_mat - mean_X) / std_X
    return {
        "mean": mean_X,
        "std": std_X,
        "normalized": normalized_X_mat
    }

