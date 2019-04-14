import numpy as np
import lib_clsr
import lib_clsr.utils
import lib_clsr.init
from scipy.special import softmax

class BaseClassifier:
    DEFAULT_PARAMS = {
        "lambda_": 0.0,
        "n_epochs": 40,
        "n_batch": 100,
        "eta": 0.01,
        "decay": 0.95,
        "dtype": "float32",
        "verbose": True,
        "wgt_init": {'scheme': 'random', 'std': 0.01},
        # "wgt_init": "xavier",
    }


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
        # set dim
        self.nclass = self.Y_train.shape[0]
        self.ndim = self.X_train.shape[0]

    def set_valid_data(self, X, y):
        self.X_valid, self.Y_valid = self._transfrom_data(X, y, self.dtype)
        self.has_validation = True

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

    def predict_log_proba(self, X):
        s_mat = self.W_mat.dot(np.transpose(X).astype(self.dtype)) + self.b_vec
        return s_mat

    # -------------------------------------------------
    # training
    def fit(self, X, y):
        """
        X (N, d):
        y (N,)
        """
        self._set_train_data(X, y)

        # initialize the parameters
        if self.wgt_init == "xavier" or self.wgt_init['scheme'] == "xavier":
            self.W_mat, self.b_vec = lib_clsr.init.get_xavier_init(
                self.ndim, self.nclass, dtype=self.dtype)
        elif self.wgt_init['scheme'] == "random":
            self.W_mat, self.b_vec = lib_clsr.init.get_random_init(
                self.ndim, self.nclass,
                self.wgt_init['std'],
                dtype=self.dtype)
        else:
            ValueError("Wrong specification of the initialization scheme")

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
        for iter_ in range(self.n_epochs):
            # mini-batch training
            self._mini_batch_train(X_train, Y_train)

            # calcualte the cost function
            train_cost = self._compute_cost(X_train, Y_train)
            if self.has_validation:
                valid_cost = self._compute_cost(self.X_valid, self.Y_valid)
            else:
                valid_cost = None

            # print out
            if self.verbose:
                print("Iteration {:d}: train_loss = {:f}; valid_loss = {:f}; lrate = {:f}".format(
                    iter_, train_cost, valid_cost, self.lrate))

            # append the cost
            self.train_costs.append(train_cost)
            if self.has_validation:
                self.valid_costs.append(valid_cost)
            # append learning rate
            self.lrates.append(self.lrate)

            # update the learning rate
            self.lrate *= self.decay

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
            grad_W, grad_b = self._compute_grad(X_batch, Y_batch)

            # update the params
            self.W_mat = self.W_mat - self.lrate * grad_W
            self.b_vec = self.b_vec - self.lrate * grad_b
