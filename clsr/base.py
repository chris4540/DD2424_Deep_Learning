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

    def print_instance_config(self):
        # print training params
        print("-------- TRAINING PARAMS --------")
        for k in self.DEFAULT_PARAMS.keys():
            print("{}: {}".format(k, getattr(self, k)))
        print("-------- TRAINING PARAMS --------")
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

        # the shape of logits is (d, N)
        assert logits.shape[1] == X.shape[1]
        # apply softmax
        s_mat = softmax(logits, axis=0)

        # obtain the top one
        ret = np.argmax(s_mat, axis=0)
        return ret

    def train(self):
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()

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

    def L2_penalty(self):
        """
        Return the L2 penalty term

        Usage:
        >>> loss = cross_entropy(...)
        >>> loss += weight_decay * model.L2_penalty()
        >>> print(loss)
        """
        ret = 0.0
        for W_mat in self.W_mats:
            ret += np.sum(W_mat**2)
        return ret

    # @staticmethod
    # def cross_entropy2(logits, labels):
    #     """
    #     calculate cross entropy with onehot
    #     """
    #     n_data = labels.shape[0]
    #     onehot_labels = utils.one_hot(labels)
    #     p_mat = softmax(logits, axis=0)
    #     p_true = np.sum(onehot_labels * p_mat, axis=0)
    #     loss = -np.log(p_true)
    #     ret = np.sum(loss) / n_data
    #     return ret
