import unittest
import numpy as np
from numpy.testing import assert_allclose
from onelayer_ann import OneLayerNetwork
from load_batch import load_batch

class OneLayerNetworkNumGrad(OneLayerNetwork):

    def compute_grad_fwd_diff(self, X_mat, Y_mat):
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


class TestOneLayerNetworkGradientCal(unittest.TestCase):

    def setUp(self):
        ndim_test = 20
        n_test_data = 10
        test_cat = 10

        self.ann = OneLayerNetworkNumGrad(test_cat, ndim_test, lambda_=0.0)
        data = load_batch("cifar-10-batches-py/data_batch_1")

        self.X_mat = data["pixel_data"][:ndim_test, :n_test_data]
        self.Y_mat = data["onehot_labels"][:, :n_test_data]

    def test_grad_fwd_diff(self):
        grad_W, grad_b = self.ann.compute_grad(self.X_mat, self.Y_mat)
        grad_W2, grad_b2 = self.ann.compute_grad_fwd_diff(self.X_mat, self.Y_mat)
        assert_allclose(grad_W, grad_W2, rtol=1e-05, atol=1e-06)
        assert_allclose(grad_b, grad_b2, rtol=1e-05, atol=1e-06)



if __name__ == '__main__':
    unittest.main()
