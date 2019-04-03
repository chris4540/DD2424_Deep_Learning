"""
One layer network
"""
import numpy as np

class OneLayerNetwork:

    def __init__(self, nclass, ndim, lambda_=None):
        """
        Args:
            nclass: the number of classes
            ndim: the size of the input vector (X)
        """
        if lambda_ is None:
            self.lambda_ = 0.01
        else:
            self.lambda_ = lambda_

        self.nclass = nclass
        self.ndim = ndim

        self.init_param()

    def init_param(self):
        sigma = 0.01 ** 2

        self.W_mat = sigma * np.random.randn(self.nclass, self.ndim)
        self.b_vec = sigma * np.random.randn(self.nclass, 1)

    def train(self):
        pass

    def evaluate(self, X_mat):
        """
        X_mat: The data, X_mat.shape == (ndim, ndata)
        """
        s_mat = self.W_mat.dot(X_mat) + self.b_vec
        p_mat = self.softmax(s_mat, axis=0)
        return p_mat

    @staticmethod
    def softmax(x, axis=None):
        return np.exp(x)/np.sum(np.exp(x), axis)

    def compute_cost(self, X_mat, Y_mat):
        n_data = X_mat.shape[1]

        # get the cross-entropy term
        p_mat = self.evaluate(X_mat)

        cross_entro = -np.log(np.sum(Y_mat*p_mat, axis=0))

        ret = (np.sum(cross_entro) / n_data) + self.get_regular_term()
        return ret

    def get_regular_term(self):
        """
        """
        return self.lambda_ * np.sum(self.W_mat**2)

    def compute_accuracy(self, X_mat, y_val):
        """
        """
        p_mat = self.evaluate(X_mat)
        y_pred = np.argmax(p, axis=0)

        ret = (y_pred == y_val).mean()
        return ret

    def compute_grad(self, X_mat, Y_mat):
        """
        Return:
            grad_W: shape = (nclass, ndim)
            grad_b: shape = (nclass, 1)
        """
        n_data = X_mat.shape[1]
        k = self.nclass
        # nclass x n_data
        p_mat = self.evaluate(X_mat)
        assert p_mat.shape == (k, n_data)
        assert Y_mat.shape == (k, n_data)
        g_mat = -(Y_mat - p_mat)

        # G * 1_{n_b} / n_b: take mean over axis 1
        grad_b = np.mean(g_mat, axis=1)
        assert grad_b.shape == (k,)
        grad_b.reshape((k, 1))

        grad_W = g_mat.dot(X_mat.T) / n_data
        grad_W += 2 * self.lambda_ * self.W_mat

        return (grad_W, grad_b)

    def compute_grads_num(self, X_mat, Y_mat):
        """
        Translated from matlab version of ComputeGradsNum
        """
        h = 1e-6
        nclass = self.nclass

        ndim = self.ndim

        grad_W = np.zeros(self.W_mat.shape)

        grad_b = np.zeros((nclass, 1))

        cost = self.compute_cost(X_mat, Y_mat);

        b = np.copy(self.b_vec)
        W = np.copy(self.W_mat)


        for i in range(nclass):
            b_old = self.b_vec[i, 0]

            self.b_vec[i, 0] = self.b_vec[i, 0] + h
            new_cost = self.compute_cost(X_mat, Y_mat)
            grad_b[i, 0] = (new_cost - cost) / h

            self.b_vec[i, 0] = b_old

        self.b_vec = np.copy(b)

        for idx in np.ndindex(W.shape):
            w_old = self.W_mat[idx]

            self.W_mat[idx] = self.W_mat[idx] + h
            new_cost = self.compute_cost(X_mat, Y_mat)
            grad_W[idx] = (new_cost - cost) / h

            self.W_mat[idx] = w_old

        self.W_mat = np.copy(W)

        return (grad_W, grad_b)
