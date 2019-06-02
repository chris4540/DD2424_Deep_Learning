"""
nn_2l.py: Two layer neural network

The network only gives the subroutines for calculate the foward propagation
and backward propagation.

No data processing and training details in this class

The class design is inspired by pyTorch

"""
import numpy as np
from scipy.special import softmax
import utils
from .base import BaseNetwork


class TwoLayerNeuralNetwork(BaseNetwork):

    DEFAULT_PARAMS = {
        "dtype": "float32",
        "verbose": True,
        "wgt_init": "xavier",
        "p_dropout": 0.0,
        "n_features": 3072,
        "n_classes": 10,
        "n_hidden_nodes": [50]
    }

    def __init__(self, **params):

        if len(params['n_hidden_nodes']) != 1:
            raise ValueError("The length of the list of hidden nodes must be 1")

        self.set_params(**params)
        if self.verbose:
            self.print_instance_config()
        self.training = True

        # init params
        self.initalize_wgts()

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

        for W_mat, b_vec in zip(self.W_mats, self.b_vecs):
            # Linear layer; z = w^{T} x + b
            z_mat = W_mat.dot(h_mat) + b_vec
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

    def backward(self, logits, labels, weight_decay=0.0):
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

    def update(self, grads, lrate):
        """
        Update the weight given the gradient
        Args:
            grads
            lrate (float): the learning rate

        Notes:
            The update eqn is:

        """
        grad_Ws, grad_bs = grads
        n_layers = len(self.W_mats)
        # update the params
        for l in range(n_layers):
            self.W_mats[l] = self.W_mats[l] - lrate * grad_Ws[l]
            self.b_vecs[l] = self.b_vecs[l] - lrate * grad_bs[l]