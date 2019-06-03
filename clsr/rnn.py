"""
For assignment 4
"""
from .base import BaseNetwork
import numpy as np
from scipy.special import softmax

class VanillaRNN(BaseNetwork):
    """
    Reference:
    torch.nn.rnn
    """

    DEFAULT_PARAMS = {
        "dtype": "float32",
        "verbose": True,
        "n_features": 80,
        "n_classes": 80,
        "seq_length": 25,
        "n_hidden_node": 200,
        "init_sigma": 0.01,
        # learning rate
        "eta": 0.1,
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
        n_class = self.n_classes

        # Eqn1: a_t = W h_{t-1} + U x_{t} + b
        # RNN.U; control the weight of input flow
        self.input_wgts = np.random.randn(m, K).astype(self.dtype) * sig
        # RNN.W; control the weight of "memory" flow
        self.hidden_wgts = np.random.randn(m, m).astype(self.dtype) * sig
        # RNN.b; bias of eqn 1; the bias term in the RNN cell
        self.cell_bias = np.zeros((m, 1), dtype=self.dtype)

        # Eqn 3: output layer
        # RNN.V; control the weight of output flow
        self.output_wgt = np.random.randn(n_class, m).astype(self.dtype) * sig
        # RNN.c
        self.output_bias = np.zeros((n_class, 1), dtype=self.dtype)


    def forward(self, inputs, h_0=None):
        """
        """
        # Do translation
        m = self.n_hidden_node
        K = self.n_features
        n_class = self.n_classes
        W = self.hidden_wgts
        U = self.input_wgts
        b = self.cell_bias
        V = self.output_wgt
        c = self.output_bias
        # ==============================================================

        inputs = inputs.astype(self.dtype)
        n_steps = inputs.shape[1]
        # make ret
        ret = np.zeros((n_class, n_steps), dtype=self.dtype)

        if h_0 is None:
            h_0 = np.zeros((m, 1), dtype=self.dtype)

        # do rnn evaluation one by one
        h_t = h_0
        for t in range(n_steps):
            x_t = inputs[:, t][:, np.newaxis]
            a_t = W.dot(h_t) + U.dot(x_t) + b
            h_t = np.tanh(a_t)
            o_t = V.dot(h_t) + c
            p_t = softmax(o_t)
            ret[:, [t]] = p_t

        return ret


    def synthesize_seq(self, x_0, h_0=None, length=5):
        """
        Consider the RNN as a generative model.
        Generate a peice of seq according to the current parameters

        Args:
            x0
            lenght (int)
        """
        # Translation
        m = self.n_hidden_node
        K = self.n_features
        n_class = self.n_classes


        if h_0 is None:
            h_0 = np.zeros((m, 1), dtype=self.dtype)

        # init h_t and x_t
        inputs = np.zeros((K, length))
        inputs[:, 0] = x_0
        outs = self.forward(inputs, h_0=h_0)

        # draw the result one by one
        ret = list()
        for t in range(length):
            prob = outs[:, t]
            draw = np.random.choice(n_class, p=prob)
            ret.append(draw)

        return ret
