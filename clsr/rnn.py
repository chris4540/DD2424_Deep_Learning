"""
For assignment 4
"""
from .base import BaseNetwork
import numpy as np

class VanillaRNN(BaseNetwork):

    DEFAULT_PARAMS = {
        "dtype": "float32",
        "verbose": True,
        "n_features": 80,
        "n_classes": 10,
        "seq_length": 25,
        "n_hidden_node": 100,
        "init_sigma": 0.01,
    }

    def __init__(self, **params):

        self.set_params(**params)

        if self.verbose:
            self.print_instance_config()

        self.initalize_wgts()

    def initalize_wgts(self):
        # make it consistent with the assignment pdf
        sig = self.init_sigma
        m = self.n_hidden_node
        K = self.n_features

        # Eqn1: a_t = W h_{t-1} + U x_{t} + b
        # RNN.U; control the weight of input flow
        self.input_wgts = np.random.randn(m, K).astype(self.dtype) * sig
        # RNN.W; control the weight of "memory" flow
        self.hidden_wgts = np.random.randn(m, m).astype(self.dtype) * sig
        # RNN.b; bias of eqn 1; the bias term in the RNN cell
        self.cell_bias = np.zeros((m, 1), dtype=self.dtype)

        # Eqn 3: output layer
        # RNN.V; control the weight of output flow
        self.output_wgt = np.random.randn(m, K).astype(self.dtype) * sig
        # RNN.c
        self.output_bias = np.zeros((K, 1), dtype=self.dtype)


    def forward(self, inputs):
        pass

