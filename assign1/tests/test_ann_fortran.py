import lib_ann
import lib_ann.ann as ann_py
import lib_ann.ann_fortran as ann_fortran
import unittest
import numpy as np
from numpy.testing import assert_allclose

class TestANNFortran(unittest.TestCase):

    def test_softmax(self):
        arr = np.random.rand(3, 2)
        assert_allclose(ann_py.softmax(arr, 0), ann_fortran.softmax(arr, 0))
        assert_allclose(ann_py.softmax(arr, 1), ann_fortran.softmax(arr, 1))

