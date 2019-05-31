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
    }

class StandardScaler:
    """
    Mimic sklearn.preprocessing.StandardScaler
    """

    def __init__(self):
        pass

    def fit(self, X_mat):
        """
        Args:
            X_mat (ndarray): the data to evalute the mean and std
                The shape of X_mat is (d, N); d is the dimension and N is the
                number of data
        """
        normalization_params = normalize_data(X_mat)
        self.mean = normalization_params['mean']
        self.std = normalization_params['std']
        # do some checking
        assert self.mean.shape[0] == X_mat.shape[0]
        assert self.std.shape[0] == X_mat.shape[0]


    def transform(self, X_mat):
        ret = (X_mat - self.mean) / self.std
        return ret

