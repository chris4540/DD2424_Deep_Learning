"""
The base class for the 2-layer network as well as the k-layer network
for Assign2 bonus and Assign 3
"""
import numpy as np
from scipy.special import softmax
from lib_clsr.init import get_xavier_init
import utils

class BaseNetwork:
    """
    The base class for 2 layer and k layer network
    """

    def __call__(self, inputs):
        return self.forward(inputs)
    # ------------------------------------------------
    # Basic utils section
    def get_params(self, deep=False):
        ret = dict()
        for k in self.DEFAULT_PARAMS.keys():
            ret[k] = getattr(self, k)
        return ret

    def set_params(self, **params):
        for k in self.DEFAULT_PARAMS.keys():
            val = params.get(k, self.DEFAULT_PARAMS[k])
            setattr(self, k, val)
        return self
    # End basic utils section
    # ------------------------------------------------
    # initialize
    def initalize_wgts(self):
        # initialize the parameters
        self.W_mats = list()
        self.b_vecs = list()
        layer_dims = [self.n_features, *self.n_hidden_nodes, self.n_classes]
        if self.wgt_init == "xavier":
            for in_dim, out_dim in utils.window(layer_dims, n=2):
                W_mat, b_vec = get_xavier_init(in_dim, out_dim, dtype=self.dtype)
                self.W_mats.append(W_mat)
                self.b_vecs.append(b_vec)
        else:
            raise ValueError("Wrong specification of the initialization scheme")
        print("Weightings and bias are initialized with %s method." % self.wgt_init)

    def predict(self, X):
        """
        Args:
            X (ndarray): the shape of the input matrix X is (d, N)
                where d is the dimension of the input
                      N is the batch size
        Return:
            the predicted classes
        """
        # get the logits from the network
        logits = self.forward(X)

        # the shape of logits is (nclasses, N)
        assert logits.shape[1] == X.shape[1]
        # apply softmax
        s_mat = softmax(logits, axis=0)

        # obtain the top one
        ret = np.argmax(s_mat, axis=0)
        return ret