import lib_ann
import lib_ann.ann as ann_py
import lib_ann.ann_f as ann_fortran
import unittest
import numpy as np
from numpy.testing import assert_allclose

class TestANNFortran(unittest.TestCase):

    def test_softmax(self):
        arr = np.random.rand(3, 2)
        assert_allclose(ann_py.softmax(arr, 0), ann_fortran.softmax(arr, 0))
        assert_allclose(ann_py.softmax(arr, 1), ann_fortran.softmax(arr, 1))

    def test_evaluate_classifier(self):
        k = 12
        d = 1000
        n = 100
        W_mat = np.random.rand(k ,d) * np.random.randint(5)
        X_mat = np.random.rand(d ,n) * np.random.randint(2)
        b_vec = np.random.rand(k ,1) * np.random.randint(3)

        sol1 = ann_py.evaluate_classifier(X_mat, W_mat, b_vec)
        sol2 = ann_fortran.evaluate_classifier(X_mat, W_mat, b_vec)

        assert_allclose(sol1, sol2)
