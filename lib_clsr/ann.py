"""
Python/Numpy implementation of ANN related fucntions
"""
import scipy
from scipy.special import softmax
import numpy as np

def evaluate_classifier(X_mat, W_mat, b_vec):
    s_mat = W_mat.dot(X_mat) + b_vec
    p_mat = softmax(s_mat, axis=0)
    return p_mat

def compute_cost(X_mat, Y_mat, W_mat, b_vec, lambda_):
    n_data = X_mat.shape[1]

    # get the cross-entropy term
    p_mat = evaluate_classifier(X_mat, W_mat, b_vec)
    cross_entro = -np.log(np.sum(Y_mat*p_mat, axis=0))

    ret = (np.sum(cross_entro) / n_data) + lambda_*np.sum(W_mat**2)
    return ret

def compute_accuracy(X_mat, y_val, W_mat, b_vec):
    p_mat = evaluate_classifier(X_mat, W_mat, b_vec)
    y_pred = np.argmax(p_mat, axis=0)
    ret = (y_pred == y_val).mean()
    return ret

def compute_gradients(X_mat, Y_mat, W_mat, b_vec, lambda_):
    n_data = X_mat.shape[1]
    # k = W_mat.shape[0]

    # p_mat.shape == (nclass, n_data)
    p_mat = evaluate_classifier(X_mat, W_mat, b_vec)

    g_mat = -(Y_mat - p_mat)

    # G * 1_{n_b} / n_b: take mean over axis 1
    grad_b = np.mean(g_mat, axis=1)[:, np.newaxis]
    # grad_b = grad_b.reshape((k, 1))

    grad_W = g_mat.dot(X_mat.T) / n_data
    grad_W += 2 * lambda_ * W_mat

    return (grad_W, grad_b)
# =============================================================================
def eval_clsr_klayers(X_mat, W_mats, b_vecs):
    """
    Args:
        W_mats (list(numpy.ndarray))

    Returns:
        cost (float): the loss or the cost of the network
        h_mats (list(np.ndarray)): the list of hidden layer values.
            Notes that the h[0] is the input layer
    """
    h_mat = X_mat
    h_mats = [h_mat]
    for W_mat, b_vec in zip(W_mats, b_vecs):
        s_mat = W_mat.dot(h_mat) + b_vec
        h_mat = np.maximum(s_mat, 0)
        #
        h_mats.append(h_mat)

    # calculate the prob.
    p_mat = softmax(s_mat, axis=0)
    return p_mat, h_mats

def compute_loss_klayers(X_mat, Y_mat, W_mats, b_vecs):
    n_data = X_mat.shape[1]
    p_mat, _ = eval_clsr_klayers(X_mat, W_mats, b_vecs)
    cross_entro = -np.log(np.sum(Y_mat*p_mat, axis=0))
    ret = (np.sum(cross_entro) / n_data)
    return ret

def compute_cost_klayers(X_mat, Y_mat, W_mats, b_vecs, lambda_):
    ret = compute_loss_klayers(X_mat, Y_mat, W_mats, b_vecs)
    for W_mat in W_mats:
        ret += lambda_*np.sum(W_mat**2)
    return ret

def compute_grads_klayers(X_mat, Y_mat, W_mats, b_vecs, lambda_):
    """
    Notes:
    The computational graph and the naming convention is:
    x(h[0]) -> s[1] -> h[1] -> s[2] -> h[2] -> ... -> h[l-1] -> s[l] -> p -> loss
    """
    n_data = X_mat.shape[1]
    n_layer = len(W_mats)

    # forward pass
    p_mat, h_mats = eval_clsr_klayers(X_mat, W_mats, b_vecs)

    grad_Ws = [None] * len(W_mats)
    grad_bs = [None] * len(b_vecs)
    # ===============================================
    # start to compute the gradient
    g_mat = -Y_mat + p_mat

    for l in range(n_layer, 0, -1):
        h_mat = h_mats[l-1]
        W_mat = W_mats[l-1] # W_mats = [W_1, W_2]; W_i = W_mats[i-1]
        #
        grad_b = np.mean(g_mat, axis=1)[:, np.newaxis]
        grad_W = g_mat.dot(h_mat.T) / n_data
        grad_W += 2 * lambda_ * W_mat

        grad_Ws[l-1] = grad_W
        grad_bs[l-1] = grad_b

        # update g_mat
        if l != 1: # do not need to update at the last loop
            g_mat = W_mat.T.dot(g_mat)
            g_mat = g_mat * (h_mat > 0)


    return grad_Ws, grad_bs
