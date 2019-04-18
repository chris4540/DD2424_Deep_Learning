import lib_clsr
import lib_clsr.ann
import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy.special import softmax

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

class TestANNTwoLayersFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = 300
        m = 50
        k = 12
        n = 10

        cls.W_mat1 = np.random.randn(m ,d).astype(np.float32) * np.sqrt(1.0/d)
        cls.W_mat2 = np.random.randn(k ,m).astype(np.float32) * np.sqrt(1.0/m)
        cls.b_vec1 = np.random.randn(m ,1).astype(np.float32)
        cls.b_vec2 = np.random.randn(k ,1).astype(np.float32)
        # b2 = np.zeros((k, 1), dtype=np.float32)
        cls.X_mat = np.random.randn(d ,n).astype(np.float32)
        cls.Y_mat = np.eye(k)[np.random.choice(k, n)].T.astype(np.float32)

    def test_2d_layer_cost_function(self):
        W_mats = [self.W_mat1, self.W_mat2]
        b_vecs = [self.b_vec1, self.b_vec2]
        for lambda_ in [0.0, 0.1, 0.5]:
            cost1 = lib_clsr.ann.compute_cost_klayer(
                self.X_mat, self.Y_mat, W_mats, b_vecs, lambda_)
            cost2 = cost_2l(
                self.X_mat, self.Y_mat, self.W_mat1, self.W_mat2,
                self.b_vec1, self.b_vec2, lambda_)
            self.assertEqual(cost1, cost2)
