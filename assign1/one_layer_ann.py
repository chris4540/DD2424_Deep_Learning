"""
One layer network for multi-linear classifier

TODO:
    rewrite to fit sckit-learn style for easy cooperate with
"""
import numpy as np
import lib_ann


class OneLayerNetwork:

    # mini-batch gradient descent params
    DEFAULT_PARAMS = {
        "lambda_": 0.0,
        "n_epochs": 40,
        "n_batch": 100,
        "eta": 0.01,
        "decay": 0.95
    }

    # variance for init paramters (W and b)
    sigma = 0.01 ** 2

    W_mat = None
    b_vec = None

    # for saving the history of training
    train_costs = None
    valid_costs = None
    lrates = None

    #
    _verbose = True

    def __init__(self, verbose=True):
        self._verbose = verbose
        self.train_costs = list()
        self.valid_costs = list()
        self.lrates = list()
        self.set_params()

    def init_weighting(self):
        self.W_mat = self.sigma * np.random.randn(self.nclass, self.ndim)
        self.b_vec = self.sigma * np.random.randn(self.nclass, 1)

    def set_train_data(self, X_train, Y_train):
        self.ndim = X_train.shape[0]
        self.nclass = Y_train.shape[0]

        # copy the training set
        self.X_train = X_train
        self.Y_train = Y_train

    def set_valid_data(self, X_valid, Y_valid):
        self.X_valid = X_valid
        self.Y_valid = Y_valid

    def get_params(self):
        ret = dict()
        for k in self.DEFAULT_PARAMS.keys():
            ret[k] = getattr(self, k)

        return ret

    def set_params(self, **params):
        for k in self.DEFAULT_PARAMS.keys():
            val = params.get(k, self.DEFAULT_PARAMS[k])
            setattr(self, k, val)
        return self

    def train(self):
        self.init_weighting()

        if self._verbose:
            # print training params
            print("-------- TRAINING PARAMS --------")
            for k in self.DEFAULT_PARAMS.keys():
                print("{}: {}".format(k, getattr(self, k)))
            print("-------- TRAINING PARAMS --------")


        X_train = self.X_train
        Y_train = self.Y_train
        self.lrate = self.eta

        for iter_ in range(self.n_epochs):

            # mini-batch training
            self._mini_batch_train(X_train, Y_train)

            # calcualte the cost function
            train_cost = self.compute_cost(X_train, Y_train)
            valid_cost = self.compute_cost(self.X_valid, self.Y_valid)

            # print out
            if self._verbose:
                print("Iteration {:d}: train_loss = {:f}; valid_loss = {:f}; lrate = {:f}".format(
                    iter_, train_cost, valid_cost, self.lrate))

            # append the cost
            self.lrates.append(self.lrate)
            self.train_costs.append(train_cost)
            self.valid_costs.append(valid_cost)
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
            grad_W, grad_b = self.compute_grad(X_batch, Y_batch)

            # update the params
            self.W_mat = self.W_mat - self.lrate * grad_W
            self.b_vec = self.b_vec - self.lrate * grad_b

    def evaluate(self, X_mat):
        """
        X_mat: The data, X_mat.shape == (ndim, ndata)
        """
        return lib_ann.evaluate_classifier(X_mat, self.W_mat, self.b_vec)

    def compute_cost(self, X_mat, Y_mat):
        ret = lib_ann.compute_cost(
            X_mat, Y_mat, self.W_mat, self.b_vec, self.lambda_)
        return ret

    def compute_accuracy(self, X_mat, y_val):
        """
        """
        return lib_ann.compute_accuracy(X_mat, y_val, self.W_mat, self.b_vec)

    def compute_grad(self, X_mat, Y_mat):
        """
        Return:
            grad_W: shape = (nclass, ndim)
            grad_b: shape = (nclass, 1)
        """
        grad_W, grad_b = lib_ann.compute_gradients(
            X_mat, Y_mat, self.W_mat, self.b_vec, self.lambda_)
        return grad_W, grad_b
