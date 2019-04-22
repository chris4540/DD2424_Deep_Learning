# from ._base import BaseClassifier
import numpy as np
import lib_clsr
import lib_clsr.ann
import lib_clsr.init
from lib_clsr.init import get_xavier_init
from scipy.special import softmax
import numpy as np
import copy
from utils.lrate import cyc_lrate
import lib_clsr.utils
from utils.preprocess import normalize_data
from scipy.special import softmax


class TwoLayerNetwork:

    size_hidden = 50
    DEFAULT_PARAMS = {
        "lambda_": 0.0,
        "n_epochs": 40,
        "n_batch": 100,
        "eta": 0.01,
        "decay": 1.0,
        "dtype": "float32",
        "verbose": True,
        "wgt_init": "xavier",
        "shuffle_per_epoch": False,
        "stop_overfit": True
    }
    _has_valid_data = False

    def __init__(self, **params):
        self.set_params(**params)

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
   # ------------------------------------------------
    # data handling section
    def _set_train_data(self, X, y):
        # set data
        self.X_train, self.Y_train = self._transfrom_data(X, y, self.dtype)
        self.y_lbl_train = y

        # set dim
        self.nclass = self.Y_train.shape[0]
        self.ndim = self.X_train.shape[0]

        # normalize the data
        data = normalize_data(self.X_train)
        self.X_mean = data['mean']
        self.X_std = data['std']
        self.X_train = data['normalized']

        if self._has_valid_data:
            self.X_valid = (self.X_valid - self.X_mean) / self.X_std

    def set_valid_data(self, X, y):
        self.X_valid, self.Y_valid = self._transfrom_data(X, y, self.dtype)
        self._has_valid_data = True
        self.y_lbl_valid = y

    @staticmethod
    def _transfrom_data(X, y, dtype=np.float32):
        """
        X (N, d):
        y (N,)
        """
        # X_mat.shape is (d, N)
        X_mat = np.transpose(X).astype(dtype)

        # Y_mat.shape is (k, N)
        Y_mat = lib_clsr.utils.conv_y_to_onehot_mat(y).astype(dtype)
        return X_mat, Y_mat
    # End data handling section
    # ------------------------------------------------
    def score(self, X, y):
        y_pred = self.predict(X)
        ret = (y_pred == y).mean()
        return ret

    def predict(self, X):
        """
        X (N, d):
        """
        s_mat = self.predict_proba(X)
        ret = np.argmax(s_mat, axis=0)
        return ret

    def predict_proba(self, X):
        # s_mat: unnormalized log probability
        s_mat = self.predict_log_proba(X)
        prob = softmax(s_mat, axis=0)
        return prob
    # =========================================================================
    def _compute_cost(self, X_mat, Y_mat):
        ret = lib_clsr.ann.compute_cost_klayers(
            X_mat, Y_mat, self.W_mats, self.b_vecs, self.lambda_)
        return ret

    def _compute_grad(self, X_mat, Y_mat):
        """
        """
        grad_Ws, grad_bs = lib_clsr.ann.compute_grads_klayers(
            X_mat, Y_mat, self.W_mats, self.b_vecs, self.lambda_)
        return grad_Ws, grad_bs

    def predict_log_proba(self, X):
        X_mat = (np.transpose(X).astype(self.dtype) - self.X_mean) / self.X_std
        p_mat, _ = lib_clsr.ann.eval_clsr_klayers(
            X_mat, self.W_mats, self.b_vecs)
        s_mat = softmax(p_mat, axis=0)
        return s_mat


    # initialize
    def _initalize_wgts(self):
        # initialize the parameters
        if self.wgt_init == "xavier" or self.wgt_init['scheme'] == "xavier":
            W_mat1, b_vec1 = get_xavier_init(self.ndim, self.size_hidden, dtype=self.dtype)
            W_mat2, b_vec2 = get_xavier_init(self.size_hidden, self.nclass, dtype=self.dtype)
            self.W_mats = [W_mat1, W_mat2]
            self.b_vecs = [b_vec1, b_vec2]
        else:
            raise ValueError("Wrong specification of the initialization scheme")

    # -------------------------------------------------
    # training
    def fit(self, X, y):
        """
        X (N, d):
        y (N,)
        """
        self._set_train_data(X, y)

        self._initalize_wgts()

        #
        self.train_costs = list()
        self.valid_costs = list()
        self.lrates = list()

        if self.verbose:
            # print training params
            print("-------- TRAINING PARAMS --------")
            for k in self.DEFAULT_PARAMS.keys():
                print("{}: {}".format(k, getattr(self, k)))
            print("-------- TRAINING PARAMS --------")

        # initialize the learning rate
        n_data = self.X_train.shape[1]
        lrates = cyc_lrate(np.arange(self.n_epochs*n_data // self.n_batch), eta_min=1e-5, eta_max=1e-1, step_size=500)
        # self.lrate = self.eta

        X_train = self.X_train
        Y_train = self.Y_train

        iter_ = 0
        train_cost = self._compute_cost(X_train, Y_train)
        valid_cost = self._compute_cost(self.X_valid, self.Y_valid)
        self.train_costs.append(train_cost)
        self.valid_costs.append(valid_cost)
        for epoch_cnt in range(self.n_epochs):

            # mini-batch training
            for j in range(n_data // self.n_batch):
                lrate = lrates[iter_]
                j_s = j*self.n_batch
                j_e = (j+1)*self.n_batch
                X_batch = X_train[:, j_s:j_e]
                Y_batch = Y_train[:, j_s:j_e]

                # get the gradient of W_mat and b_vec
                grad_Ws, grad_bs = self._compute_grad(X_batch, Y_batch)

                # update the params
                for l in range(len(self.W_mats)):
                    self.W_mats[l] = self.W_mats[l] - lrate * grad_Ws[l]
                    self.b_vecs[l] = self.b_vecs[l] - lrate * grad_bs[l]

                iter_ += 1
            # =============================================================

            # calcualte
            train_cost = self._compute_cost(X_train, Y_train)
            # train_acc = self.score(X_train, self.y_lbl_train)
            if self._has_valid_data:
                valid_cost = self._compute_cost(self.X_valid, self.Y_valid)
                # valid_acc = self.score(self.X_valid, )
            else:
                valid_cost = 0.0



            # append the cost
            self.train_costs.append(train_cost)
            if self._has_valid_data:
                self.valid_costs.append(valid_cost)

            # print out
            if self.verbose:
                print("Epoch {:d}: Iteration {:d}: train_loss = {:f};"
                        " valid_loss = {:f}; lrate = {:f}".format(
                        epoch_cnt, iter_, train_cost, valid_cost, lrate))


            # check if training cost
            if train_cost < 1e-6:
                break

    def _is_valid_cost_going_up(self):
        if len(self.valid_costs) < 2:
            return False

        if self.shuffle_per_epoch:
            ret = self.valid_costs[-1] > np.mean(self.valid_costs[-10:-2])
        else:
            ret = self.valid_costs[-1] > self.valid_costs[-2]
        return ret

    def _mini_batch_train(self, X_train, Y_train):
        """
        Perform Mini batch gradient descent for one epoch
        """
        # train with mini-batch
        n_data = X_train.shape[1]
        for j in range(n_data // self.n_batch):
            j_s = j*self.n_batch
            j_e = (j+1)*self.n_batch
            X_batch = X_train[:, j_s:j_e]
            Y_batch = Y_train[:, j_s:j_e]

            # get the gradient of W_mat and b_vec
            grad_Ws, grad_bs = self._compute_grad(X_batch, Y_batch)

            # update the params
            for l in range(len(self.W_mats)):
                self.W_mats[l] = self.W_mats[l] - self.lrate * grad_Ws[l]
                self.b_vecs[l] = self.b_vecs[l] - self.lrate * grad_bs[l]
            # self.W_mat = self.W_mat - self.lrate * grad_W
            # self.b_vec = self.b_vec - self.lrate * grad_b
