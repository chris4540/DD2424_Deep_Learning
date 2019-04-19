import lib_clsr
import lib_clsr.ann
import unittest
import numpy as np
from tests.utils import compute_grad_klayers_fwd_diff
from tests.utils import compute_grad_klayers_cent_diff
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from scipy.special import softmax
from utils.load_batch import load_batch
from utils.preprocess import normalize_data

def cost_2l(X_mat, Y_mat, W_mat1, W_mat2, b_vec1, b_vec2, lambda_):
    n_data = X_mat.shape[1]
    s_1 = W_mat1.dot(X_mat) + b_vec1
    h_1 = np.maximum(0, s_1)
    s_2 = W_mat2.dot(h_1) + b_vec2
    p_mat = softmax(s_2, axis=0)
    cross_entro = -np.log(np.sum(Y_mat*p_mat, axis=0))
    #
    ret = (np.sum(cross_entro) / n_data)
    ret += lambda_*np.sum(W_mat1**2)
    ret += lambda_*np.sum(W_mat2**2)
    return ret

def grad_2l(X_mat, Y_mat, W_mat1, W_mat2, b_vec1, b_vec2, lambda_):
    n_data = X_mat.shape[1]
    s_1 = W_mat1.dot(X_mat) + b_vec1
    h_1 = np.maximum(0, s_1)
    s_2 = W_mat2.dot(h_1) + b_vec2
    p_mat = softmax(s_2, axis=0)
    # =================================================
    g_mat = -(Y_mat - p_mat)
    grad_b2 = np.mean(g_mat, axis=1)[:, np.newaxis]
    grad_W2 = g_mat.dot(h_1.T) / n_data + 2*lambda_ * W_mat2
    # =============================================================
    g_mat = (W_mat2.T).dot(g_mat)
    g_mat = g_mat * (s_1 > 0)
    # =============================================================
    grad_b1 = np.mean(g_mat, axis=1)[:, np.newaxis]
    grad_W1 = g_mat.dot(X_mat.T) / n_data + 2*lambda_*W_mat1

    return (grad_W1, grad_W2, grad_b1, grad_b2)

class TestANNTwoLayersFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = 20
        m = 15
        k = 10
        n = 5

        cls.W_mat1 = np.random.randn(m ,d).astype(np.float32) * np.sqrt(1.0/d)
        cls.W_mat2 = np.random.randn(k ,m).astype(np.float32) * np.sqrt(1.0/m)
        cls.b_vec1 = np.random.randn(m ,1).astype(np.float32)
        cls.b_vec2 = np.random.randn(k ,1).astype(np.float32)
        # b2 = np.zeros((k, 1), dtype=np.float32)
        train_data = load_batch("cifar-10-batches-py/data_batch_1")
        X_mat = train_data["pixel_data"][:d, :n]
        cls.X_mat = normalize_data(X_mat)['normalized'].astype(np.float32)
        cls.Y_mat = train_data['onehot_labels'][:k, :n].astype(np.float32)

    def test_2d_layer_cost_func(self):
        W_mats = [self.W_mat1, self.W_mat2]
        b_vecs = [self.b_vec1, self.b_vec2]
        for lambda_ in [0.0, 0.1, 0.5, 0.7, 1.0]:
            cost1 = lib_clsr.ann.compute_cost_klayers(
                self.X_mat, self.Y_mat, W_mats, b_vecs, lambda_)
            cost2 = cost_2l(
                self.X_mat, self.Y_mat, self.W_mat1, self.W_mat2,
                self.b_vec1, self.b_vec2, lambda_)
            self.assertEqual(cost1, cost2)

    def test_2d_grad_func(self):
        lambda_ = 0.0

        W_mats = [self.W_mat1, self.W_mat2]
        b_vecs = [self.b_vec1, self.b_vec2]
        for lambda_ in [0.0, 0.1, 0.5, 0.7, 1.0]:
            grad2_W1, grad2_W2, grad2_b1, grad2_b2 = grad_2l(
                self.X_mat, self.Y_mat, self.W_mat1, self.W_mat2,
                self.b_vec1, self.b_vec2, lambda_)

            grad_Ws, grad_bs = lib_clsr.ann.compute_grads_klayers(
                self.X_mat, self.Y_mat, W_mats, b_vecs, lambda_)

            # compare
            assert_array_equal(grad2_W1, grad_Ws[0])
            assert_array_equal(grad2_W2, grad_Ws[1])
            assert_array_equal(grad2_b1, grad_bs[0])
            assert_array_equal(grad2_b2, grad_bs[1])

    def test_grad_central_diff(self):
        lambda_ = 0.0
        step = 1e-5
        W_mats = [self.W_mat1, self.W_mat2]
        b_vecs = [self.b_vec1, self.b_vec2]
        [W1n, W2n], [b1n, b2n] = compute_grad_klayers_fwd_diff(
                self.X_mat, self.Y_mat, W_mats, b_vecs, lambda_, step,
                lib_clsr.ann.compute_cost_klayers)
        [W1a, W2a], [b1a, b2a] = lib_clsr.ann.compute_grads_klayers(
                self.X_mat, self.Y_mat, W_mats, b_vecs, lambda_)

        assert_allclose(W1n, W1a)
        assert_allclose(W2n, W2a)
        assert_allclose(b1n, b1a)
        assert_allclose(b2n, b2a)

    def test_grad_central_diff(self):
        lambda_ = 0.0
        step = 5e-3
        atol = 1e-3
        rtol = 1e-3
        W_mats = [self.W_mat1, self.W_mat2]
        b_vecs = [self.b_vec1, self.b_vec2]
        [W1n, W2n], [b1n, b2n] = compute_grad_klayers_cent_diff(
                self.X_mat, self.Y_mat, W_mats, b_vecs, lambda_, step,
                lib_clsr.ann.compute_cost_klayers)
        [W1a, W2a], [b1a, b2a] = lib_clsr.ann.compute_grads_klayers(
                self.X_mat, self.Y_mat, W_mats, b_vecs, lambda_)

        assert_allclose(W1n, W1a, atol=atol, rtol=rtol)
        assert_allclose(W2n, W2a, atol=atol, rtol=rtol)
        assert_allclose(b1n, b1a, atol=atol, rtol=rtol)
        assert_allclose(b2n, b2a, atol=atol, rtol=rtol)
