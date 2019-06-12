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
        self.cur_mean = None
        self.cur_var = None
        self.eps = np.finfo(dtype).eps
        self.momentum = momentum

    def __call__(self, inputs):
        return self.batch_normalize(inputs)

    def batch_normalize(self, inputs):

        if self.training:
            batch_mean = np.mean(inputs, axis=1)[:, np.newaxis]
            batch_var = np.var(inputs, axis=1)[:, np.newaxis]
            self.cur_mean = batch_mean
            self.cur_var = batch_var
            self.update_running(batch_mean, batch_var)
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var

        ret = (inputs - batch_mean) / np.sqrt(batch_var + self.eps)
        return ret

    def update_running(self, mean, var):

        # when first time
        if self.running_mean is None and self.running_var is None:
            self.running_mean = mean
            self.running_var = var
            return

        alpha = self.momentum
        beta = 1 - alpha

        # update running mean
        self.running_mean = alpha*self.running_mean + beta*mean

        # update running var
        self.running_var = alpha*self.running_var + beta*var

    def back_pass(self, g_mat, s_cap):
        """
        Implementation of BatchNormBackPass
        See asignment for details

        Enployed a different way to calculate the backpass of batchnorm layer

        Derives:
        https://kevinzakka.github.io/2016/09/14/batch_normalization/
        """

        n_data = s_cap.shape[1]

        # the dominator
        dom = (self.cur_var + self.eps)**(0.5) * n_data
        term1 = n_data * g_mat
        term2 = np.sum(g_mat, axis=1, keepdims=True)
        term3 = s_cap*np.sum(g_mat*s_cap, axis=1, keepdims=True)
        nom = term1 - term2 - term3
        ret = nom / dom
        return ret

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
        # "wgt_init": "random",
        "init_sig" : 1,
        "p_dropout": 0.0,
        "n_features": 3072,
        "n_classes": 10,
        "n_hidden_nodes": [50],
        "batch_norm": False,
        "batch_norm_momentum": 0.7,
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
            if self.batch_norm:
                self.z_caps = []  # the output of batch norm layer
                # self.z_mats = []  # the output of affine layers / fully-connected layer

        nlayer = len(self.W_mats)

        for i, (W_mat, b_vec) in enumerate(zip(self.W_mats, self.b_vecs)):
            # Linear layer; z = w^{T} x + b
            z_mat = W_mat.dot(h_mat) + b_vec

            # Apply the activation except the last layer(the logist out layer)
            if i >= nlayer-1:
                break

            # apply batch norm
            if self.batch_norm:
                # store the affine layer output
                # self.z_mats.append(z_mat)
                bn_layer = self.batch_norm_layer[i]
                z_cap = bn_layer(z_mat)
                # store the batch norm layer output
                self.z_caps.append(z_cap)
                z_mat = self.bn_scales[i] * z_cap + self.bn_shifts[i]

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
                self.batch_norm_layer.append(
                    BatchNormalizeLayer(ndim, momentum=self.batch_norm_momentum, dtype=self.dtype)
                )
                # the shifts
                self.bn_shifts.append(np.zeros((ndim, 1), dtype=self.dtype))
                # the scales
                self.bn_scales.append(np.ones((ndim, 1), dtype=self.dtype))


    def _get_backward_grad(self, logits, labels, weight_decay=0.0):
        """
        Calculate the gradients using predined backward propagation formulae

        Notes:
        The computational graph and the naming convention is:
        x(h[0]) -> s[1] -> h[1] -> s[2] -> h[2] -> ... -> h[l-1] -> s[l] -> p -> loss
        And the hidden states are saved when training

        This routine is refactored from  lib_clsr.ann.compute_grads_klayers

        Usage:
        >>> model.train()
        >>> out = model.forward(inputs)
        """
        # get the number of layers
        n_layers = len(self.W_mats)
        grad_Ws = [None] * n_layers
        grad_bs = [None] * n_layers
        #
        grad_scales = [None] * (n_layers-1)
        grad_shifts = [None] * (n_layers-1)

        # get the number datas
        n_data = labels.shape[0]
        assert logits.shape[1] == n_data

        # Propagate the gradient through the loss and softmax operations
        p_mat = softmax(logits, axis=0)
        Y_mat = utils.one_hot(labels)
        g_mat = -Y_mat + p_mat
        g_mat.astype(self.dtype)

        # calculate the last layer
        h_mat = self.hidden_output[-1]
        W_mat = self.W_mats[-1]
        grad_b = np.mean(g_mat, axis=1)[:, np.newaxis]
        grad_W = g_mat.dot(h_mat.T) / n_data
        grad_W += 2 * weight_decay * W_mat
        g_mat = W_mat.T.dot(g_mat)
        g_mat = g_mat * (h_mat > 0)
        grad_Ws[-1] = grad_W.astype(self.dtype)
        grad_bs[-1] = grad_b.astype(self.dtype)

        # ==============================================
        # calculate from k-1 to the first layer
        for l in range(n_layers-1, 0, -1):
            # ====================================================
            # back-propagation
            if self.batch_norm:
                # compute the delta scale and delta shift
                grad_shift = np.mean(g_mat, axis=1)[:, np.newaxis]
                z_cap = self.z_caps[l-1]
                grad_scale = np.mean(g_mat * z_cap, axis=1)[:, np.newaxis]

                grad_scales[l-1] = grad_scale
                grad_shifts[l-1] = grad_shift
                # ================================================================
                # Propagate the gradients through the scale and shift
                g_mat = g_mat * self.bn_scales[l-1]
                # Propagate through the batch normalization
                bn_layer = self.batch_norm_layer[l-1]
                g_mat = bn_layer.back_pass(g_mat, z_cap)

            # ====================================================
            h_mat = self.hidden_output[l-1]
            W_mat = self.W_mats[l-1] # W_mats = [W_1, W_2]; W_i = W_mats[i-1]
            #
            grad_b = np.mean(g_mat, axis=1)[:, np.newaxis]
            grad_W = g_mat.dot(h_mat.T) / n_data
            grad_W += 2 * weight_decay * W_mat

            # store the values
            grad_Ws[l-1] = grad_W.astype(self.dtype)
            grad_bs[l-1] = grad_b.astype(self.dtype)
            # ====================================================

            # update g_mat
            if l != 1: # do not need to update at the last loop
                g_mat = W_mat.T.dot(g_mat)
                g_mat = g_mat * (h_mat > 0)

        ret = {
            'grad_W_mats': grad_Ws,
            'grad_b_vecs': grad_bs
        }
        if self.batch_norm:
            ret['grad_bn_scales'] = grad_scales
            ret['grad_bn_shifts'] = grad_shifts

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
        n_layers = len(self.W_mats)

        grads = self._get_backward_grad(logits, labels, weight_decay)
        grad_Ws = grads['grad_W_mats']
        grad_bs = grads['grad_b_vecs']
        # update the params
        for l in range(n_layers):
            self.W_mats[l] = self.W_mats[l] - lrate * grad_Ws[l]
            self.b_vecs[l] = self.b_vecs[l] - lrate * grad_bs[l]

        # update the batch norm
        if self.batch_norm:
            grad_scales = grads['grad_bn_shifts']
            grad_shifts = grads['grad_bn_scales']
            for l in range(n_layers-1):
                self.bn_scales[l] = self.bn_scales[l] - lrate * grad_scales[l]
                self.bn_shifts[l] = self.bn_shifts[l] - lrate * grad_shifts[l]

    def parameters(self):
        ret = [
            'W_mats',
            'b_vecs',
        ]
        if self.batch_norm:
            ret.append("bn_shifts")
            ret.append("bn_scales")

        return ret
