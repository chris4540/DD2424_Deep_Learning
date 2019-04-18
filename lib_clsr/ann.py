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

def evaluate_classifier_klayer(X_mat, Y_mat, W_mats, b_vecs):
    """
    Args:
        W_mats (list(numpy.ndarray))
    """
    h_mat = X_mat
    h_mats = [h_mat]
    s_mats = [None]
    for W_mat, b_vec in zip(W_mats, b_vecs):
        s_mat = W_mat.dot(h_mat) + b_vec
        h_mat = np.maximum(s_mat, 0)
        #
        h_mats.append(h_mat)
        s_mats.append(s_mat)

    # calculate the prob.
    p_mat = softmax(s_mat, axis=0)
    return p_mat, h_mats, s_mats

def compute_cost_klayer(X_mat, Y_mat, W_mats, b_vecs, lambda_):
    n_data = X_mat.shape[1]
    p_mat, _, _ = evaluate_classifier_klayer(X_mat, Y_mat, W_mats, b_vecs)
    cross_entro = -np.log(np.sum(Y_mat*p_mat, axis=0))
    cost = (np.sum(cross_entro) / n_data)
    for W_mat in W_mats:
        cost += lambda_*np.sum(W_mat)
    return cost

def compute_gradients_klayer(X_mat, Y_mat, W_mats, b_vecs, lambda_):
    n_data = X_mat.shape[1]
    n_layer = len(W_mats)

    # forward pass
    p_mat, h_mats, s_mats = evaluate_classifier_klayer(X_mat, Y_mat, W_mats, b_vecs)

    grad_Ws = [None] * len(W_mats)
    grad_bs = [None] * len(b_vecs)
    # ===============================================
    # start to compute the gradient
    g_mat = -(Y_mat - p_mat)

    for l in range(n_layer, 0, -1):
        h_mat = h_mats[l-1]
        s_mat = s_mats[l-1]
        W_mat = W_mats[l-1]
        #
        grad_b = np.mean(g_mat, axis=1)[:, np.newaxis]
        grad_W = g_mat.dot(h_mat.T) / n_data
        grad_W += 2 * lambda_ * W_mat

        grad_Ws[l-1] = grad_W
        grad_bs[l-1] = grad_b

        # update g_mat
        if l != 1:
            g_mat = W_mat.T.dot(g_mat)
            g_mat = g_mat * (s_mat > 0)


    return grad_Ws, grad_bs