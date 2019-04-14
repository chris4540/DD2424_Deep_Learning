import numpy as np
from ._base import BaseClassifier
import lib_clsr
import lib_clsr.svm


class SupportVectorMachine(BaseClassifier):

    DEFAULT_PARAMS = {
        "lambda_": 0.0,
        "n_epochs": 40,
        "n_batch": 100,
        "eta": 0.01,
        "decay": 0.95,
        "dtype": "float32",
        "verbose": True,
    }

    verbose = True

    def __init__(self, **params):
        self.set_params(**params)

    # def predict_proba(self, X):
    #     neg_log_prob = self.predict_log_proba(X)
    #     prob = softmax(s_mat, axis=0)
    #     return prob

    # def predict_log_proba(self, X):
    #     s_mat = self.W_mat.dot(np.transpose(X).astype(self.dtype)) + self.b_vec
    #     return s_mat

    # =========================================================================
    def _compute_cost(self, X_mat, Y_mat):
        ret = lib_clsr.svm.compute_cost(
            X_mat, Y_mat, self.W_mat, self.b_vec, self.lambda_)
        return ret

    def _compute_grad(self, X_mat, Y_mat):
        """
        Return:
            grad_W: shape = (nclass, ndim)
            grad_b: shape = (nclass, 1)
        """
        grad_W, grad_b = lib_clsr.svm.compute_gradients(
            X_mat, Y_mat, self.W_mat, self.b_vec, self.lambda_)
        return grad_W, grad_b
