import numpy as np
from ._base import BaseClassifier
import lib_clsr
import lib_clsr.svm


class SupportVectorMachine(BaseClassifier):

    verbose = True

    def __init__(self, **params):
        self.set_params(**params)
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
