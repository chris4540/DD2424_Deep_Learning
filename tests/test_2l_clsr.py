from utils.load_batch import load_batch
from clsr.two_layer_network import TwoLayerNetwork
from time import time
import unittest


class TestANNTwoLayersFunction(unittest.TestCase):

    def test_sanity_check(self):
        train_data = load_batch("cifar-10-batches-py/data_batch_1")
        valid_data = load_batch("cifar-10-batches-py/data_batch_2")
        test_data = load_batch("cifar-10-batches-py/test_batch")
        ann = TwoLayerNetwork(stop_overfit=False, n_epochs=500, eta=0.02, verbose=False)
        # set validation data
        ann.set_valid_data(valid_data["pixel_data"][:, :100].T, valid_data["labels"][:100])
        ann.fit(train_data["pixel_data"][:, :100].T, train_data["labels"][:100])
        train_cost = ann.train_costs[-1]
        self.assertLess(train_cost, 0.3)
