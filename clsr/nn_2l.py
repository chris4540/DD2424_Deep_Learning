"""
nn_2l.py: Two layer neural network

The network only gives the subroutines for calculate the foward propagation
and backward propagation.

No data processing and training details in this class
"""
import numpy as np
from .base import BaseNetwork


class TwoLayerNeuralNetwork(BaseNetwork):

    size_hidden = 50
    DEFAULT_PARAMS = {
        "dtype": "float32",
        "verbose": True,
        "wgt_init": "xavier",
        "dropout": False,
        "n_features": 100,
        "n_classes": 10,
        "n_hidden_nodes": [50]
    }

    _has_valid_data = False

    def __init__(self, **params):
        self.set_params(**params)
        self.initalized = False
        if self.verbose:
            self.print_instance_config()

    def print_instance_config(self):
        # print training params
        print("-------- TRAINING PARAMS --------")
        for k in self.DEFAULT_PARAMS.keys():
            print("{}: {}".format(k, getattr(self, k)))
        print("-------- TRAINING PARAMS --------")


    def forward(self, X_mat):
        """
        Return the logits (unnormalized log probability)
        """
        pass