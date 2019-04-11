import lib_ann
from lib_ann import py_ann_f
from lib_ann import ann as ann_py
import unittest
import numpy as np
from numpy.testing import assert_allclose
from load_batch import load_batch

class TestANNFortran(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = load_batch("cifar-10-batches-py/data_batch_1")

    def test_softmax(self):
        arr = np.random.rand(3, 2)
        assert_allclose(ann_py.softmax(arr, 0), py_ann_f.ann_f.softmax(arr, 0))
        assert_allclose(ann_py.softmax(arr, 1), py_ann_f.ann_f.softmax(arr, 1))

    def test_evaluate_classifier(self):
        k = 12
        d = 1000
        n = 100
        W_mat = np.random.randn(k ,d)
        X_mat = np.random.rand(d ,n) * np.random.randint(2)
        b_vec = np.random.randn(k ,1)

        sol1 = ann_py.evaluate_classifier(X_mat, W_mat, b_vec)
        sol2 = py_ann_f.ann_f.evaluate_classifier(X_mat, W_mat, b_vec)

        assert_allclose(sol1, sol2)

    def test_compute_cost(self):
        k = self.data["onehot_labels"].shape[0]
        d = self.data["pixel_data"].shape[0]
        W_mat = np.random.randn(k ,d)
        b_vec = np.random.randn(k ,1)

        X_mat = self.data["pixel_data"][:, :]
        Y_mat = self.data["onehot_labels"][:, :]
        sol1 = ann_py.compute_cost(X_mat, Y_mat, W_mat, b_vec, 0.1)
        sol2 = py_ann_f.ann_f.compute_cost(X_mat, Y_mat, W_mat, b_vec, 0.1)
        assert_allclose(sol1, sol2)

    def test_compute_grad(self):
        k = self.data["onehot_labels"].shape[0]
        d = self.data["pixel_data"].shape[0]
        W_mat = np.random.randn(k ,d)
        b_vec = np.random.randn(k ,1)

        X_mat = self.data["pixel_data"][:, :]
        Y_mat = self.data["onehot_labels"][:, :]
        grad_W1, grad_b1 = ann_py.compute_gradients(X_mat, Y_mat, W_mat, b_vec, 0.1)

        #
        grad_W2, grad_b2 = py_ann_f.ann_f.compute_gradients(X_mat, Y_mat, W_mat, b_vec, 0.1)
        assert_allclose(grad_W1, grad_W2)
        assert_allclose(grad_b1, grad_b2)
