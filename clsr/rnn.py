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
        "n_hidden_node": 100,
        "init_sigma": 0.01,
        # learning rate
        # "eta": 0.1,
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
        n_data = logits.shape[1]
        p = softmax(logits, axis=0)
        p_true = p[labels, range(n_data)]
        log_likelihood = -np.log(p_true)
        loss = np.sum(log_likelihood)
        return loss

    def forward(self, input_seq, h_0=None):
        inputs = self._get_one_hot(input_seq)
        inputs = inputs.astype(self.dtype)
        if h_0 is None:
            h_0 = np.zeros((self.n_hidden_node, 1), dtype=self.dtype)
        return self._forward(inputs, h_0)

    def _forward(self, inputs, h_0):
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
        h_t = np.copy(h_0)
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

        return ret, h_t


    def _get_backward_grad(self, logits, targets, clipping=True):
        """
        Args:
            logits: shape == (K, T)
            targets: shape == (T,)
        """
        labels_oh = self._get_one_hot(targets)
        nsteps = logits.shape[1]
        # do some translation
        U = self.input_wgts
        W = self.hidden_wgts
        b = self.cell_bias
        V = self.output_wgt
        c = self.output_bias

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
            'grad_hidden_wgts': grad_W,
            'grad_input_wgts': grad_U,
            'grad_cell_bias': grad_b,
            'grad_output_wgt': grad_V,
            'grad_output_bias': grad_c
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

    def _get_one_hot(self, part_seq):
        one_idx = np.array(part_seq)
        nkind = self.n_features
        nlabels = len(part_seq)
        ret = np.zeros((nkind, nlabels))
        ret[one_idx, np.arange(nlabels)] = 1
        return ret

    def synthesize_seq(self, input_, h_0, length=5):
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

        # x_0 = self._get_one_hot(inputs)

        # draw the result one by one
        x = input_
        h = np.copy(h_0)
        ret = list()
        for t in range(length):
            out, h = self.forward([x], h)
            prob = softmax(out).flatten()
            x = np.random.choice(n_class, p=prob)
            ret.append(x)

        return ret

    def parameters(self):
        ret = [
            'hidden_wgts',
            'cell_bias',
            'input_wgts',
            'output_wgt',
            'output_bias'
        ]
        return ret

    def state_dict(self):
        params = self.parameters()
        ret = dict()
        for attr in params:
            theta = getattr(self, attr)  # obtain the parameter reference
            ret[attr] = np.copy(theta)

        return ret

    def load_state_dict(self, state_dict):
        for k in state_dict.keys():
            setattr(self, k, state_dict[k])
