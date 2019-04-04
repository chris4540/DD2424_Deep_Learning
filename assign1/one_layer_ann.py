"""
One layer network
"""
import numpy as np

class OneLayerNetwork:

    n_batch = 100
    eta = 0.01
    n_epochs = 40

    def __init__(self, nclass, ndim, lambda_=0.0):
        """
        Args:
            nclass: the number of classes
            ndim: the size of the input vector (X)
        """
        self.lambda_ = lambda_

        self.nclass = nclass
        self.ndim = ndim

        self.init_param()

    def init_param(self):
        sigma = 0.01 ** 2

        self.W_mat = sigma * np.random.randn(self.nclass, self.ndim)
        self.b_vec = sigma * np.random.randn(self.nclass, 1)

    def set_train_data(self, X_train, Y_train):
        # copy the training set
        self.X_train = np.copy(X_train)
        self.Y_train = np.copy(Y_train)

    def set_valid_data(self, X_valid, Y_valid):
        pass

    def train(self):


        for iter_ in range(self.n_epochs):
            # shuffle the samples
            if True:
                idx = np.random.rand(self.X_train.shape[1]).argsort()
                X_train = np.take(self.X_train, idx, axis=1)
                Y_train = np.take(self.Y_train, idx, axis=1)
            else:
                X_train = self.X_train
                Y_train = self.Y_train

            # mini-batch training
            self._mini_batch_train(X_train, Y_train)

            # calcualte the cost function
            cost = self.compute_cost(X_train, Y_train)
            print("Iteration %d: Cost = %f" % (iter_, cost))

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

        ret = (np.sum(cross_entro) / n_data) + self.get_regular_term()
        return ret

    def get_regular_term(self):
        """
        """
        return self.lambda_ * np.sum(self.W_mat**2)

    def compute_accuracy(self, X_mat, y_val):
        """
        """
        p_mat = self.evaluate(X_mat)
        y_pred = np.argmax(p, axis=0)

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
        assert p_mat.shape == (k, n_data)
        assert Y_mat.shape == (k, n_data)
        g_mat = -(Y_mat - p_mat)

        # G * 1_{n_b} / n_b: take mean over axis 1
        grad_b = np.mean(g_mat, axis=1)
        assert grad_b.shape == (k,)
        grad_b = grad_b.reshape((k, 1))

        grad_W = g_mat.dot(X_mat.T) / n_data
        grad_W += 2 * self.lambda_ * self.W_mat

        return (grad_W, grad_b)
