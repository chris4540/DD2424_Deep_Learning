import unittest
import numpy as np
import lib_clsr.ann
import utils
from utils.load_batch import load_batch
from lib_clsr.utils import conv_y_to_onehot_mat
from tests.utils import compute_grad_klayers_cent_diff
from numpy.testing import assert_allclose

class TestKLayersGradients(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        n_data = 10
        img_dim = 20
        train_data = load_batch("cifar-10-batches-py/data_batch_1")
        cls.X_mat = train_data["pixel_data"][:img_dim, :n_data]
        cls.Y_mat = conv_y_to_onehot_mat(train_data["labels"])[:, :n_data]
        cls.dtype = np.float64
        cls.in_dim = cls.X_mat.shape[0]
        cls.out_dim = cls.Y_mat.shape[0]

    def get_inits(self, layer_dims):
        W_mats = []
        b_vecs = []
        for in_dim, out_dim in utils.window(layer_dims, n=2):
            W_mat = np.random.randn(out_dim, in_dim).astype(self.dtype) * np.sqrt(1.0/in_dim)
            b_vec = np.random.randn(out_dim ,1).astype(self.dtype)
            W_mats.append(W_mat)
            b_vecs.append(b_vec)

        return W_mats, b_vecs

    def check_close_central_grad(self, W_mats, b_vecs):
        step = 1e-5
        atol = 1e-05
        rtol = 1e-07

        for lambda_ in [0.0, 0.1, 0.7, 1.0]:
            gradW_num, gradb_num = compute_grad_klayers_cent_diff(
                    self.X_mat, self.Y_mat, W_mats, b_vecs, lambda_, step,
                    lib_clsr.ann.compute_cost_klayers)

            gradW_anl, gradb_anl = lib_clsr.ann.compute_grads_klayers(
                self.X_mat, self.Y_mat, W_mats, b_vecs, lambda_)

            for Wn, Wa in zip(gradW_num, gradW_anl):
                assert_allclose(Wn, Wa, atol=atol, rtol=rtol)

            for bvec_n, bvec_a in zip(gradb_num, gradb_anl):
                assert_allclose(bvec_n, bvec_a, atol=atol, rtol=rtol)

    def test_4_layer_grad(self):
        layer_dims = [self.in_dim, 50, 30, 20, self.out_dim]
        W_mats, b_vecs = self.get_inits(layer_dims)
        self.check_close_central_grad(W_mats, b_vecs)

    def test_3_layer_grad(self):
        layer_dims = [self.in_dim, 50, 30, self.out_dim]
        W_mats, b_vecs = self.get_inits(layer_dims)
        self.check_close_central_grad(W_mats, b_vecs)

    def test_2_layer_grad(self):
        layer_dims = [self.in_dim, 50, self.out_dim]
        W_mats, b_vecs = self.get_inits(layer_dims)
        self.check_close_central_grad(W_mats, b_vecs)