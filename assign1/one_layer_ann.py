"""
One layer network
"""
import numpy as np

class OneLayerNetwork:

    # mini-batch gradient descent params
    N_BATCH_DEFAULT = 100
    ETA_DEFAULT = 0.01
    N_EPOCHS_DEFAULT = 10

    # variance for init paramters (W and b)
    sigma = 0.01 ** 2

    W_mat = None
    b_vec = None

    # for saving the history of training
    train_costs = None
    valid_costs = None

    #
    _verbose = True

    def __init__(self, nclass, ndim, lambda_=0.0, verbose=True):
        """
        Initialize the model with parameters

        Args:
            nclass: the number of classes
            ndim: the size of the input vector (X)
            lambda_: the weight of the regularization term
        """
        # set network parameters
        self.nclass = nclass
        self.ndim = ndim
        # set hyperparameter: the weight of the regularization term
        self.lambda_ = lambda_

        # set if verbose
        self._verbose = verbose

        if self._verbose:
            print("-------- MODEL PARAMS --------")
            for k in ["nclass", "ndim", "lambda_"]:
                print("{}: {}".format(k, getattr(self, k)))
            print("-------- MODEL PARAMS --------")

        # initialize classifier parameters
        self.init_param()
        # set the default values for the training parameters
        self.set_train_params()

        self.train_costs = list()
        self.valid_costs = list()

    def init_param(self):

        self.W_mat = self.sigma * np.random.randn(self.nclass, self.ndim)
        self.b_vec = self.sigma * np.random.randn(self.nclass, 1)

    def set_train_data(self, X_train, Y_train):
        # copy the training set
        self.X_train = np.copy(X_train)
        self.Y_train = np.copy(Y_train)

    def set_valid_data(self, X_valid, Y_valid):
        self.X_valid = np.copy(X_valid)
        self.Y_valid = np.copy(Y_valid)

    def set_train_params(self, *args, **kwargs):
        """
        Set the training parameters
        """
        self.n_batch = kwargs.get("n_batch", self.N_BATCH_DEFAULT)
        self.n_epochs = kwargs.get("n_epochs", self.N_EPOCHS_DEFAULT)
        self.eta = kwargs.get("eta", self.ETA_DEFAULT)

    def train(self):

        if self._verbose:
            # print training params
            print("-------- TRAINING PARAMS --------")
            for k in ["n_batch", "n_epochs", "eta"]:
                print("{}: {}".format(k, getattr(self, k)))
            print("-------- TRAINING PARAMS --------")

        X_train = self.X_train
        Y_train = self.Y_train

        for iter_ in range(self.n_epochs):
            # shuffle the samples
            if iter_!= 0 and iter_ % 10 == 0:
                # idx = np.random.rand(self.X_train.shape[1]).argsort()
                idx = np.arange(X_train.shape[1])
                np.random.shuffle(idx)
                X_train = np.take(self.X_train, idx, axis=1)
                Y_train = np.take(self.Y_train, idx, axis=1)

            # mini-batch training
            self._mini_batch_train(X_train, Y_train)

            # calcualte the cost function
            train_cost = self.compute_cost(X_train, Y_train)
            valid_cost = self.compute_cost(self.X_valid, self.Y_valid)

            # print out
            if self._verbose:
                print("Iteration {:d}: train_loss = {:f}; valid_loss = {:f}".format(
                    iter_, train_cost, valid_cost))

            # append the cost
            self.train_costs.append(train_cost)
            self.valid_costs.append(valid_cost)

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
            self.W_mat = self.W_mat - self.eta * grad_W
            self.b_vec = self.b_vec - self.eta * grad_b

    def evaluate(self, X_mat):
        """
        X_mat: The data, X_mat.shape == (ndim, ndata)
        """
        s_mat = self.W_mat.dot(X_mat) + self.b_vec
        p_mat = self.softmax(s_mat, axis=0)
        return p_mat

    @staticmethod
    def softmax(x, axis=None):
        return np.exp(x)/np.sum(np.exp(x), axis)

    def compute_cost(self, X_mat, Y_mat):
        n_data = X_mat.shape[1]

        # get the cross-entropy term
        p_mat = self.evaluate(X_mat)

        cross_entro = -np.log(np.sum(Y_mat*p_mat, axis=0))

        ret = (np.sum(cross_entro) / n_data) + self.lambda_*np.sum(self.W_mat**2)
        return ret

    def compute_accuracy(self, X_mat, y_val):
        """
        """
        p_mat = self.evaluate(X_mat)
        y_pred = np.argmax(p_mat, axis=0)

        ret = (y_pred == y_val).mean()
        return ret

    def compute_grad(self, X_mat, Y_mat):
        """
        Return:
            grad_W: shape = (nclass, ndim)
            grad_b: shape = (nclass, 1)
        """
        n_data = X_mat.shape[1]
        k = self.nclass
        # nclass x n_data
        p_mat = self.evaluate(X_mat)
        # assert p_mat.shape == (k, n_data)
        # assert Y_mat.shape == (k, n_data)
        g_mat = -(Y_mat - p_mat)

        # G * 1_{n_b} / n_b: take mean over axis 1
        grad_b = np.mean(g_mat, axis=1)
        # assert grad_b.shape == (k,)
        grad_b = grad_b.reshape((k, 1))

        grad_W = g_mat.dot(X_mat.T) / n_data
        grad_W += 2 * self.lambda_ * self.W_mat

        return (grad_W, grad_b)