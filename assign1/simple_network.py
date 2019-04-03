"""
One layer network
"""
import numpy as np

class OneLayerNetwork:

    def __init__(self, nclass, ndim):
        """
        Args:
            nclass: the number of classes
            ndim: the size of the input vector (X)
        """
        self.nclass = nclass
        self.ndim = ndim

        self.init_param()

    def init_param(self):
        sigma = 0.01 ** 2

        self.W_mat = sigma * np.random.randn(self.nclass, self.ndim)
        self.b_vec = sigma * np.random.randn(self.nclass, 1)

        self.lambda_ = 0.01

    def train(self):
        pass

    def evaluate(self, X_mat):
        """
        X_mat: The data, X_mat.shape == (ndim, ndata)
        """
        s_mat = self.W_mat.dot(X_mat) + self.b_vec
        p_mat = self.softmax(s_mat)
        return p_mat

    @staticmethod
    def softmax(x):
        return np.exp(x)/np.sum(np.exp(x))

    def compute_cost(self, X_mat, y_mat):
        # get the cross-entropy term
        p_mat = self.evaluate(X_mat)

        cross_entro = -np.log(np.sum(y_mat*p_mat, axis=0))

        ret = np.sum(cross_entro) + self.get_regular_term()
        return ret

    def get_regular_term(self):
        return self.lambda_ * np.sum(self.W_mat**2)

    def computte_accuracy(self):
        pass
