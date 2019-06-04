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
        self.training = True

    def __call__(self, inputs, init_hidden=None):
        return self.forward(inputs, init_hidden)

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

    @staticmethod
    def cross_entropy(logits, labels):
        """
        Return the cross entropy loss
        """
        n_data = labels.shape[0]
        p = softmax(logits, axis=0)
        p_true = p[labels, range(n_data)]
        log_likelihood = -np.log(p_true)
        loss = np.sum(log_likelihood)
        return loss

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
        if h_0 is None:
            h_0 = np.zeros((m, 1), dtype=self.dtype)

        inputs = inputs.astype(self.dtype)
        n_steps = inputs.shape[1]
        # make ret
        ret = np.zeros((n_class, n_steps), dtype=self.dtype)

        # store hidden states
        if self.training:
            self.h_vec_time = np.zeros((m, n_steps), dtype=self.dtype)
            self.a_vec_time = np.zeros((m, n_steps), dtype=self.dtype)
            self.x_vec_time = np.zeros((K, n_steps), dtype=self.dtype)
            self.h_0 = h_0

        # do rnn evaluation one by one
        h_t = h_0
        for t in range(n_steps):
            x_t = inputs[:, t][:, np.newaxis]
            a_t = W.dot(h_t) + U.dot(x_t) + b
            h_t = np.tanh(a_t)
            o_t = V.dot(h_t) + c
            ret[:, [t]] = o_t
            # record down
            if self.training:
                self.h_vec_time[:, [t]] = h_t
                self.a_vec_time[:, [t]] = a_t
                self.x_vec_time[:, [t]] = x_t

        return ret


    def _get_backward_grad(self, logits, labels_oh, clipping=False):
        """
        Args:
            logits: shape == (K, T)
            labels_oh: shape == (K, T)
        """
        nsteps = logits.shape[1]
        # do some translation
        U = self.input_wgts
        W = self.hidden_wgts
        V = self.output_wgt
        c = self.output_bias
        b = self.cell_bias

        # calculate the gradient back pro throught softmax and
        p_mat_T = softmax(logits, axis=0)
        g_mat_T = -labels_oh + p_mat_T  # over time
        g_mat_T = g_mat_T.astype(self.dtype)

        # back to the output layer; similar to nn_kl, but not taking batch mean
        grad_V = g_mat_T.dot(self.h_vec_time.T)
        assert grad_V.shape == V.shape
        grad_c = np.sum(g_mat_T, axis=1, keepdims=True)
        assert grad_c.shape == c.shape
        # =============================================================
        # back propagate at last time step
        # calculate dL/dh_{T}
        dLdh = np.zeros(self.h_vec_time.shape, dtype=self.dtype)
        dLdh[:, -1] = g_mat_T[:, -1].dot(V)
        # calculate dL/da_{T}
        dLda = np.zeros(self.h_vec_time.shape, dtype=self.dtype)
        dLda[:, -1] = dLdh[:, -1].dot(np.diag(1 - np.tanh(self.a_vec_time[:, -1])**2))

        # loop over time
        for t in range(nsteps-2, -1, -1):
            # print(t)
            dLdh[:, t] = g_mat_T[:, t].dot(V) + dLda[:, t+1].dot(W)
            dLda[:, t] = dLdh[:, t].dot(np.diag(1 - np.tanh(self.a_vec_time[:, t])**2))

        # calculate grad U
        grad_U = dLda.dot(self.x_vec_time.T)
        assert grad_U.shape == U.shape
        # calculate grad b
        grad_b = np.sum(dLda, axis=1, keepdims=True)
        assert grad_b.shape == b.shape
        # calculate grad W
        # build a vectors from t=0 to t=T-1
        h_vecs_shift = np.hstack((self.h_0, self.h_vec_time[:, :-1]))
        grad_W = dLda.dot(h_vecs_shift.T)
        assert grad_W.shape == W.shape

        # make the grad result as a dictionary
        ret = {
            'grad_W': grad_W,
            'grad_b': grad_b,
            'grad_U': grad_U,
            'grad_V': grad_V,
            'grad_c': grad_c
        }

        if clipping:
            ret = self.clipping(ret)
        return ret

    def clipping(self, grads):
        min_ = -5
        max_ = 5
        ret = dict()
        for k in grads.keys():
            ret[k] = np.clip(grads[k], min_, max_)
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
            prob = softmax(outs[:, t])
            draw = np.random.choice(n_class, p=prob)
            ret.append(draw)

        return ret
