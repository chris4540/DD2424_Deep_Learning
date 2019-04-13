from tests.utils import compute_grad_central_diff
from tests.utils import compute_grad_fwd_diff
import lib_clsr
import lib_clsr.ann
import unittest
import numpy as np
from numpy.testing import assert_allclose


class TestSVMFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        k = 12
        d = 200
        n = 50
        cls.W_mat = np.random.randn(k ,d)
        cls.X_mat = np.random.randn(d ,n)
        cls.b_vec = np.random.randn(k ,1)
        cls.Y_mat = np.eye(k)[np.random.choice(k, n)].T
        cls.lambda_ = 0.01

    def test_grad_central_diff(self):
        cost_func = lib_clsr.ann.compute_cost

        # compute gradient with analytical method
        grad_W, grad_b = lib_clsr.ann.compute_gradients(
            self.X_mat, self.Y_mat, self.W_mat, self.b_vec, self.lambda_)

        # compute gradient with central different
        grad_W_numric, grad_b_numric = compute_grad_central_diff(
            self.X_mat, self.Y_mat, self.W_mat, self.b_vec, self.lambda_, cost_func)

        # test them are close
        assert_allclose(grad_b, grad_b_numric, rtol=1e-07, atol=1e-07)
        assert_allclose(grad_W, grad_W_numric, rtol=1e-07, atol=1e-07)

    def test_grad_fwd_diff(self):
        cost_func = lib_clsr.ann.compute_cost

        # compute gradient with analytical method
        grad_W, grad_b = lib_clsr.ann.compute_gradients(
            self.X_mat, self.Y_mat, self.W_mat, self.b_vec, self.lambda_)

        # compute gradient with central different
        grad_W_numric, grad_b_numric = compute_grad_fwd_diff(
            self.X_mat, self.Y_mat, self.W_mat, self.b_vec, self.lambda_, cost_func)

        # test them are close
        assert_allclose(grad_b, grad_b_numric, rtol=1e-06, atol=1e-07)
        assert_allclose(grad_W, grad_W_numric, rtol=1e-06, atol=1e-07)
