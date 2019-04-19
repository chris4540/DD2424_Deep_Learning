import lib_clsr
from lib_clsr import ann_f
from lib_clsr import ann as ann_py
import unittest
import numpy as np
from numpy.testing import assert_allclose
from time import time
# alias
ann_f = ann_f.ann_for

class TestANNFortran(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        k = 12
        d = 4000
        n = 500
        dtype = 'float32'
        cls.W_mat = np.random.randn(k ,d).astype(dtype) * np.sqrt(1.0/d)
        cls.X_mat = np.random.randn(d ,n).astype(dtype) * np.random.randint(2)
        cls.b_vec = np.random.randn(k ,1).astype(dtype)
        cls.Y_mat = np.eye(k)[np.random.choice(k, n)].T.astype(dtype)
        cls.lambda_ = 0.01

    def test_softmax(self):
        arr = self.W_mat.dot(self.X_mat) + self.b_vec
        assert_allclose(ann_py.softmax(arr, 0), ann_f.softmax(arr, 0), atol=1e-6)
        assert_allclose(ann_py.softmax(arr, 1), ann_f.softmax(arr, 1), atol=1e-6)

    def test_softmax_performance(self):
        N = 100
        arr = self.W_mat.dot(self.X_mat) + self.b_vec
        st = time()
        for _ in range(N):
            ann_py.softmax(arr, 0)
        py_time = time() - st

        st = time()
        for _ in range(N):
            ann_f.softmax(arr, 0)
        fortran_time = time() - st
        self.assertLess(fortran_time, py_time)

    def test_evaluate_classifier(self):
        sol1 = ann_py.evaluate_classifier(self.X_mat, self.W_mat, self.b_vec)
        sol2 = ann_f.evaluate_classifier(self.X_mat, self.W_mat, self.b_vec)
        assert_allclose(sol1, sol2, atol=1e-6)

    def test_compute_cost(self):

        sol1 = ann_py.compute_cost(self.X_mat, self.Y_mat, self.W_mat, self.b_vec, self.lambda_)
        sol2 = ann_f.compute_cost(self.X_mat, self.Y_mat, self.W_mat, self.b_vec, self.lambda_)
        assert_allclose(sol1, sol2, atol=1e-6)

    def test_compute_grad(self):
        grad_W1, grad_b1 = ann_py.compute_gradients(self.X_mat, self.Y_mat, self.W_mat, self.b_vec, self.lambda_)
        grad_W2, grad_b2 = ann_f.compute_gradients(self.X_mat, self.Y_mat, self.W_mat, self.b_vec, self.lambda_)
        assert_allclose(grad_W1, grad_W2, atol=1e-6)
        assert_allclose(grad_b1, grad_b2, atol=1e-6)
