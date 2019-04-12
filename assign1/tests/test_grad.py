import unittest
import numpy as np
from numpy.testing import assert_allclose
from one_layer_ann import OneLayerNetwork
from load_batch import load_batch

class OneLayerNetworkNumGrad(OneLayerNetwork):
    """
    Helper class to insert numerical method to compute gradient
    """

    def compute_grad_fwd_diff(self, X_mat, Y_mat):
        """
        Translated from matlab version of ComputeGradsNum
        """
        h = 1e-6
        h_inv = 1.0 / h
        nclass = self.nclass
        ndim = self.ndim
        grad_W = np.zeros(self.W_mat.shape)
        grad_b = np.zeros((nclass, 1))

        cost = self.compute_cost(X_mat, Y_mat);

        for i in range(nclass):
            b_old = self.b_vec[i, 0]

            self.b_vec[i, 0] = b_old + h
            new_cost = self.compute_cost(X_mat, Y_mat)
            grad_b[i, 0] = (new_cost - cost) * h_inv

            self.b_vec[i, 0] = b_old

        for idx in np.ndindex(self.W_mat.shape):
            w_old = self.W_mat[idx]

            self.W_mat[idx] = w_old + h
            new_cost = self.compute_cost(X_mat, Y_mat)
            grad_W[idx] = (new_cost - cost) * h_inv

            self.W_mat[idx] = w_old

        return (grad_W, grad_b)

    def compute_grad_central_diff(self, X_mat, Y_mat):
        """
        Translated from matlab version of ComputeGradsNum
        """
        h = 1e-6
        h_inv = 1.0 / h
        nclass = self.nclass
        ndim = self.ndim
        grad_W = np.zeros(self.W_mat.shape)
        grad_b = np.zeros((nclass, 1))

        # cost = self.compute_cost(X_mat, Y_mat);

        for i in range(nclass):
            b_old = self.b_vec[i, 0]

            self.b_vec[i, 0] = b_old + h
            c1 = self.compute_cost(X_mat, Y_mat)

            self.b_vec[i, 0] = b_old - h
            c2 = self.compute_cost(X_mat, Y_mat)

            grad_b[i, 0] = (c1 - c2) * h_inv * 0.5

            self.b_vec[i, 0] = b_old

        for idx in np.ndindex(self.W_mat.shape):
            w_old = self.W_mat[idx]

            self.W_mat[idx] = w_old + h
            c1 = self.compute_cost(X_mat, Y_mat)

            self.W_mat[idx] = w_old - h
            c2 = self.compute_cost(X_mat, Y_mat)

            grad_W[idx] = (c1 - c2) * h_inv * 0.5

            self.W_mat[idx] = w_old

        return (grad_W, grad_b)


class TestOneLayerNetworkGradientCal(unittest.TestCase):

    def setUp(self):
        n_test_data = 10
        ndim_small = 20
        ndim = 500  # not use the full dimension as it takes a long time

        data = load_batch("cifar-10-batches-py/data_batch_1")
        self.X_mat_small = data["pixel_data"][:ndim_small, :n_test_data]
        self.X_mat = data["pixel_data"][:ndim, :n_test_data]
        self.Y_mat = data["onehot_labels"][:, :n_test_data]

        nclass = self.Y_mat.shape[0]

        self.ann_small = OneLayerNetworkNumGrad()
        self.ann_small.ndim = ndim_small
        self.ann_small.nclass = nclass
        self.ann_small.init_weighting()

        self.ann = OneLayerNetworkNumGrad()
        self.ann.ndim = ndim
        self.ann.nclass = nclass
        self.ann.init_weighting()

    def test_grad_fwd_diff_small(self):
        X_mat = self.X_mat_small
        Y_mat = self.Y_mat

        ann = self.ann_small
        self.check_all_close_fwd_diff(ann, X_mat, Y_mat)

    def test_grad_cen_diff_small(self):
        X_mat = self.X_mat_small
        Y_mat = self.Y_mat

        ann = self.ann_small
        self.check_all_close_central_diff(ann, X_mat, Y_mat)

    def test_grad_fwd_diff(self):
        X_mat = self.X_mat
        Y_mat = self.Y_mat
        ann = self.ann

        self.check_all_close_fwd_diff(ann, X_mat, Y_mat)

    def test_grad_cen_diff(self):
        X_mat = self.X_mat
        Y_mat = self.Y_mat
        ann = self.ann

        self.check_all_close_central_diff(ann, X_mat, Y_mat)

    @staticmethod
    def check_all_close_fwd_diff(network, X_mat, Y_mat):
        grad_W, grad_b = network.compute_grad(X_mat, Y_mat)
        grad_W2, grad_b2 = network.compute_grad_fwd_diff(X_mat, Y_mat)

        # since foward different method is a first order approx.
        # release the restriction for all_close method
        assert_allclose(grad_W, grad_W2, rtol=1e-06, atol=1e-07)
        assert_allclose(grad_b, grad_b2, rtol=1e-06, atol=1e-07)

    @staticmethod
    def check_all_close_central_diff(network, X_mat, Y_mat):
        grad_W, grad_b = network.compute_grad(X_mat, Y_mat)
        grad_W2, grad_b2 = network.compute_grad_central_diff(X_mat, Y_mat)
        assert_allclose(grad_W, grad_W2, rtol=1e-07, atol=1e-09)
        assert_allclose(grad_b, grad_b2, rtol=1e-07, atol=1e-09)


if __name__ == '__main__':
    unittest.main()
