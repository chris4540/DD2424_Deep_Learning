"""
Batch normalization:
https://medium.com/@sh.tsang/review-batch-normalization-inception-v2-bn-inception-the-2nd-to-surpass-human-level-18e2d0f56651
http://violin-tao.blogspot.com/2018/02/ml-batch-normalization.html
"""
import numpy as np
from scipy.special import softmax
import utils
from .base import BaseNetwork
from .nn_2l import TwoLayerNeuralNetwork

class BatchNormalizeLayer:
    """
    A wrapper of the batch normalization realted parameters

    Simple class to:
        1. batch normalize the input mini-batch
        2. update the moving avg

    Args:
        ndims (int): the input dimension. Expected invariant to the batch size
        momentum (float): the alpha value in the assignment
            the value used for the running_mean and running_var computation.
            The statistic update eqn is:
                x_update = momentum * x_est + (1-momentum)*x_obs
    """

    def __init__(self, ndims, momentum=0.9, dtype='float32'):
        # a flag to mark if we need to update the
        self.training = True
        self.running_mean = None
        self.running_var = None
        self.eps = np.finfo(dtype).eps
        self.momentum = momentum

    def __call__(self, inputs):
        return self.batch_normalize(inputs)

    def batch_normalize(self, inputs):
        batch_mean = np.mean(inputs, axis=1)[:, np.newaxis]
        batch_var = np.var(inputs, axis=1)[:, np.newaxis]

        ret = (inputs - batch_mean) / np.sqrt(batch_var + self.eps)

        if self.training:
            self.update_running(batch_mean, batch_var)
        return ret

    def update_running(self, mean, var):
        alpha = self.momentum
        beta = 1 - alpha

        # update running mean
        if self.running_mean is None:
            self.running_mean = mean
        else:
            self.running_mean = alpha*self.running_mean + beta*mean

        # update running var
        if self.running_var is None:
            self.running_var = var
        else:
            self.running_var = alpha*self.running_var + beta*var

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

# class KLayerNeuralNetwork(BaseNetwork):
class KLayerNeuralNetwork(TwoLayerNeuralNetwork):

    DEFAULT_PARAMS = {
        "dtype": "float32",
        "verbose": True,
        "wgt_init": "xavier",
        "p_dropout": 0.0,
        "n_features": 3072,
        "n_classes": 10,
        "n_hidden_nodes": [50],
        "batch_norm": True
    }

    def __init__(self, **params):
        self.set_params(**params)
        if self.verbose:
            self.print_instance_config()
        self.training = True

        # init params
        self.initalize_wgts()

    def train(self):
        self.training = True
        # turn on the batch norm layers
        if self.batch_norm:
            for l in self.batch_norm_layer:
                l.train()

    def eval(self):
        self.training = False
        # turn off the batch norm layers
        if self.batch_norm:
            for l in self.batch_norm_layer:
                l.eval()

    def forward(self, X_mat):
        """
        Return the logits (unnormalized log probability)

        Notes:
            refactored from lib_clsr.ann
        """

        h_mat = X_mat.astype(self.dtype)
        # hidden_output: the list for saved the hidden layer outputs
        if self.training:
            self.hidden_output = [h_mat]

        nlayer = len(self.W_mats)

        for i, (W_mat, b_vec) in enumerate(zip(self.W_mats, self.b_vecs)):
            # Linear layer; z = w^{T} x + b
            z_mat = W_mat.dot(h_mat) + b_vec

            # Apply the activation except the last layer(the logist out layer)
            if i >= nlayer-1:
                continue

            # apply batch norm
            if self.batch_norm:
                bn_layer = self.batch_norm_layer[i]
                z_cap = bn_layer(z_mat)
                z_mat_tmp = self.bn_scales[i] * z_cap + self.bn_shifts[i]

            # ReLU activation function / rectifier
            h_mat = np.maximum(z_mat, 0)
            # apply dropout
            if self.training and self.p_dropout > 0.0:
                mask = self.get_dropout_mask(h_mat, p=self.p_dropout)
                h_mat *= mask

            # save down the hidden layer output
            if self.training:
                self.hidden_output.append(h_mat)

        return z_mat

    def initalize_wgts(self):
        super().initalize_wgts()
        # add batch norm layers
        if self.batch_norm:
            self.batch_norm_layer = list()
            self.bn_shifts = list()
            self.bn_scales = list()
            for ndim in self.n_hidden_nodes:
                # the bn layers
                self.batch_norm_layer.append(BatchNormalizeLayer(ndim))
                # the shifts
                self.bn_shifts.append(np.zeros((ndim, 1)))
                # the scales
                self.bn_scales.append(np.ones((ndim, 1)))


    def _get_backward_grad(self, logits, labels, weight_decay=0.0):
        """
        Calculate the gradients using predined backward propagation formulas

        Notes:
        The computational graph and the naming convention is:
        x(h[0]) -> s[1] -> h[1] -> s[2] -> h[2] -> ... -> h[l-1] -> s[l] -> p -> loss
        And the hidden states are saved when training

        This routine is refactored from  lib_clsr.ann.compute_grads_klayers

        Usage:
        >>> model.train()
        >>> out = model.forward(inputs)
        >>> grads = model.backward(out, labels)
        >>> model.update(grads, lrate)
        """
        # get the number of layers
        n_layers = len(self.W_mats)
        grad_Ws = [None] * n_layers
        grad_bs = [None] * n_layers

        # get the number datas
        n_data = labels.shape[0]
        assert logits.shape[1] == n_data

        p_mat = softmax(logits, axis=0)
        Y_mat = utils.one_hot(labels)
        g_mat = -Y_mat + p_mat
        g_mat.astype(self.dtype)

        for l in range(n_layers, 0, -1):
            h_mat = self.hidden_output[l-1]
            W_mat = self.W_mats[l-1] # W_mats = [W_1, W_2]; W_i = W_mats[i-1]
            #
            grad_b = np.mean(g_mat, axis=1)[:, np.newaxis]
            grad_W = g_mat.dot(h_mat.T) / n_data
            grad_W += 2 * weight_decay * W_mat

            grad_Ws[l-1] = grad_W.astype(self.dtype)
            grad_bs[l-1] = grad_b.astype(self.dtype)

            # update g_mat
            if l != 1: # do not need to update at the last loop
                g_mat = W_mat.T.dot(g_mat)
                g_mat = g_mat * (h_mat > 0)

        ret = (grad_Ws, grad_bs)
        return ret

    def backward(self, logits, labels, weight_decay, lrate):
        """
        Update the weight given the gradient
        Args:
            grads
            lrate (float): the learning rate

        Notes:
            The update eqn is:

        """
        grad_Ws, grad_bs = self._get_backward_grad(logits, labels, weight_decay)
        n_layers = len(self.W_mats)
        # update the params
        for l in range(n_layers):
            self.W_mats[l] = self.W_mats[l] - lrate * grad_Ws[l]
            self.b_vecs[l] = self.b_vecs[l] - lrate * grad_bs[l]
