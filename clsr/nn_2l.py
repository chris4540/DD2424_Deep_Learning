"""
nn_2l.py: Two layer neural network

The network only gives the subroutines for calculate the foward propagation
and backward propagation.

No data processing and training details in this class

The class design is inspired by pyTorch

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
        assert len(params['n_hidden_nodes']) == 1
        self.set_params(**params)
        if self.verbose:
            self.print_instance_config()
        self.save_hidden_output = True

        # init params
        self.initalize_wgts()

    def print_instance_config(self):
        # print training params
        print("-------- TRAINING PARAMS --------")
        for k in self.DEFAULT_PARAMS.keys():
            print("{}: {}".format(k, getattr(self, k)))
        print("-------- TRAINING PARAMS --------")

    def train(self):
        self.save_hidden_output = True

    def eval(self):
        self.save_hidden_output = False

    def forward(self, X_mat):
        """
        Return the logits (unnormalized log probability)

        Notes:
            refactored from lib_clsr.ann

        """

        h_mat = X_mat.T
        # hidden_output: the list saved the hidden layer output
        if self.save_hidden_output:
            self.hidden_output = [h_mat]

        for W_mat, b_vec in zip(self.W_mats, self.b_vecs):
            # Linear layer; z = w^{T} x + b
            z_mat = W_mat.dot(h_mat) + b_vec
            # ReLU activation function / rectifier
            h_mat = np.maximum(z_mat, 0)
            # save down the hidden layer output
            self.hidden_output.append(h_mat)

        return z_mat.T