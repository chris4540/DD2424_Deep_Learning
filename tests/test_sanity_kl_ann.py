"""
Test the sanity of k layers network
"""
from utils.load_batch import load_batch
from clsr.k_layer_network import KLayerNetwork
from time import time
import unittest


class TestKLayersSanity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        n_data = 100
        train_data = load_batch("cifar-10-batches-py/data_batch_1")
        cls.train_img = train_data["pixel_data"][:, :n_data].T
        cls.train_labels = train_data["labels"][:n_data]

        cls.params = {
            "n_epochs": 500,
            "lambda_": 0,
            "verbose": False,
            "lrate_scheme": {
                "scheme": "cyclic",
                "eta_lim": [1e-5, 1e-1],
                "step_size": 500
            }
        }

    def test_2_layer_sanity_check(self):
        ann = KLayerNetwork(n_layers=2, n_hidden_nodes=[50], **self.params)
        ann.fit(self.train_img, self.train_labels)
        train_cost = ann.train_costs[-1]
        self.assertLess(train_cost, 0.05)

    def test_3_layer_sanity_check(self):
        ann = KLayerNetwork(n_layers=3, n_hidden_nodes=[50, 50], **self.params)
        ann.fit(self.train_img, self.train_labels)
        train_cost = ann.train_costs[-1]
        self.assertLess(train_cost, 0.05)

    def test_4_layer_sanity_check(self):
        ann = KLayerNetwork(n_layers=4, n_hidden_nodes=[50, 50, 50], **self.params)
        ann.fit(self.train_img, self.train_labels)
        train_cost = ann.train_costs[-1]
        self.assertLess(train_cost, 0.05)

