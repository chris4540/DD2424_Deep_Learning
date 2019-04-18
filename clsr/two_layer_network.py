from ._base import BaseClassifier
import lib_clsr
import lib_clsr.ann
import lib_clsr.init
from lib_clsr.init import get_xavier_init
from scipy.special import softmax
import numpy as np
import copy

class TwoLayerNetwork(BaseClassifier):

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
        p_mat, _ = lib_clsr.ann.eval_clsr_klayers(
            np.transpose(X), self.W_mats, self.b_vecs)
        s_mat = softmax(p_mat, axis=0)
        return s_mat


    # initialize
    def _initalize_wgts(self):
        print("In _initalize_wgts")
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
        self.lrate = self.eta

        X_train = self.X_train
        Y_train = self.Y_train

        n_data = X_train.shape[1]

        # initialize shuffle idx array
        if self.shuffle_per_epoch:
            shuffle_idx = np.arange(n_data)

        #
        W_mat_best = None
        b_vec_best = None
        valid_up_cnt = 0

        for iter_ in range(self.n_epochs):
            if self.shuffle_per_epoch:
                np.random.shuffle(shuffle_idx)
                X_train = np.take(self.X_train, shuffle_idx, axis=1)
                Y_train = np.take(self.Y_train, shuffle_idx, axis=1)

            # mini-batch training
            self._mini_batch_train(X_train, Y_train)

            # calcualte the cost function
            train_cost = self._compute_cost(X_train, Y_train)
            if self._has_valid_data:
                valid_cost = self._compute_cost(self.X_valid, self.Y_valid)
            else:
                valid_cost = 0.0

            # print out
            if self.verbose:
                print("Iteration {:d}: train_loss = {:f}; valid_loss = {:f}; lrate = {:f}".format(
                    iter_, train_cost, valid_cost, self.lrate))

            # append the cost
            self.train_costs.append(train_cost)
            if self._has_valid_data:
                self.valid_costs.append(valid_cost)
            # append learning rate
            self.lrates.append(self.lrate)

            # update the learning rate
            self.lrate *= self.decay

            # check if training cost
            if train_cost < 1e-6:
                break

            if self.stop_overfit and iter_ > 2 and self._is_valid_cost_going_up():
                if valid_up_cnt == 0:
                    print("Warning: the validation cost increase. Saving the weighting")
                    W_mats_best = copy.deepcopy(self.W_mats)
                    b_vecs_best = copy.deepcopy(self.b_vecs)
                elif valid_up_cnt > 5:
                    print("Warning: Overfitting, will stop the training")
                    break
                valid_up_cnt += 1
            else:
                # reset the counter due to it goes down again
                valid_up_cnt = 0
                W_mats_best = None
                b_vecs_best = None

        #
        if valid_up_cnt > 0:
            print("Notice: Updating back the best weighting")
            self.W_mats = W_mats_best
            self.b_vecs = b_vecs_best

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
