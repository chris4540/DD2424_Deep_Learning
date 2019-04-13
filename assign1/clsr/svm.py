import numpy as np
import lib_clsr
import lib_clsr.svm
import lib_clsr.utils
import lib_clsr.init

class SupportVectorMachine:

    DEFAULT_PARAMS = {
        "lambda_": 0.0,
        "n_epochs": 40,
        "n_batch": 100,
        "eta": 0.01,
        "decay": 0.95,
        "dtype": "float32",
        "verbose": True,
    }

    verbose = True

    def __init__(self, **params):
        self.set_params(**params)


    def fit(self, X, y):
        """
        X (N, d):
        y (N,)
        """
        self._set_train_data(X, y)

        # initialize the parameters
        self.W_mat, self.b_vec = lib_clsr.init.get_xavier_init(
            self.ndim, self.nclass, dtype=self.dtype)

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

        #
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

    def predict(self, X):
        """
        X (N, d):
        """
        s_mat = self.predict_log_proba(X)
        ret = np.argmax(s_mat, axis=0)
        return ret

    def predict_log_proba(self, X):
        s_mat = self.W_mat.dot(np.transpose(X).astype(self.dtype)) + self.b_vec
        return s_mat

    def score(self, X, y):
        y_pred = self.predict(X)
        ret = (y_pred == y).mean()
        return ret

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

    # ===========================================
    # data handling
    def _set_train_data(self, X, y):
        # set data
        self.X_train, self.Y_train = self._transfrom_data(X, y)
        # set dim
        self.nclass = self.Y_train.shape[0]
        self.ndim = self.X_train.shape[0]

    def set_valid_data(self, X, y):
        self.X_valid, self.Y_valid = self._transfrom_data(X, y)
        self.has_validation = True

    def _transfrom_data(self, X, y):
        """
        X (N, d):
        y (N,)
        """
        # X_mat.shape is (d, N)
        X_mat = np.transpose(X).astype(self.dtype)

        # Y_mat.shape is (k, N)
        Y_mat = lib_clsr.utils.conv_y_to_onehot_mat(y).astype(self.dtype)
        return X_mat, Y_mat
    # =========================================================================
    def _compute_cost(self, X_mat, Y_mat):
        ret = lib_clsr.svm.compute_cost(
            X_mat, Y_mat, self.W_mat, self.b_vec, self.lambda_)
        return ret

    def _compute_grad(self, X_mat, Y_mat):
        """
        Return:
            grad_W: shape = (nclass, ndim)
            grad_b: shape = (nclass, 1)
        """
        grad_W, grad_b = lib_clsr.svm.compute_gradients(
            X_mat, Y_mat, self.W_mat, self.b_vec, self.lambda_)
        return grad_W, grad_b

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
