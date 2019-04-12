"""
Python/Numpy implementation of ANN related fucntions
"""
import numpy as np
from lib_ann import softmax

def _softmax_py(x, axis=0):
    if axis == 0:
        ret = np.exp(x)/np.sum(np.exp(x), 0)
    elif axis == 1:
        ret = (np.exp(x).T/np.sum(np.exp(x), 1)).T
    else:
        raise ValueError("Not support axis > 1")
    return ret

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
    k = W_mat.shape[0]

    # p_mat.shape == (nclass, n_data)
    p_mat = evaluate_classifier(X_mat, W_mat, b_vec)

    g_mat = -(Y_mat - p_mat)

    # G * 1_{n_b} / n_b: take mean over axis 1
    grad_b = np.mean(g_mat, axis=1)[:, np.newaxis]
    # grad_b = grad_b.reshape((k, 1))

    grad_W = g_mat.dot(X_mat.T) / n_data
    grad_W += 2 * lambda_ * W_mat

    return (grad_W, grad_b)
