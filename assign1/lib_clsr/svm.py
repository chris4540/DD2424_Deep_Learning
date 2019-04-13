import numpy as np

def compute_cost(X_mat, Y_mat, W_mat, b_vec, lambda_):
    """
    \sum_{i!=y}[max(0, \vec{s} - s_y + 1)]
    """
    n_data = X_mat.shape[1]
    s_mat = W_mat.dot(X_mat) + b_vec
    s_y_vec = np.sum(s_mat * Y_mat, axis=0) # shape = (ndata,)
    hinge_mat = np.maximum(s_mat - s_y_vec + 1, 0)

    # we summed n_data extra ones. Therefore,
    # (np.sum(hinge_mat) - n_data) / ndata
    ret = (np.sum(hinge_mat)/n_data - 1) + 0.5*lambda_*np.sum(W_mat**2)
    return ret

def compute_gradients(X_mat, Y_mat, W_mat, b_vec, lambda_):
    n_data = X_mat.shape[1]
    #
    s_mat = W_mat.dot(X_mat) + b_vec
    s_y_vec = np.sum(s_mat * Y_mat, axis=0) # shape = (ndata,)

    # ------------------------
    # start making g_mat
    # ------------------------
    # hinge_mat = np.maximum(s_mat - s_y_vec + 1, 0)
    # g_mat = hinge_mat.copy()  # but we never use hinge_mat again
    g_mat = np.maximum(s_mat - s_y_vec + 1, 0)
    # for incorrect classes
    g_mat[g_mat > 0] = 1
    # for the correct classes
    g_mat[Y_mat == 1] = 0
    # compute the gradient of the correct classes
    crt_cls_grad = -np.sum(g_mat, axis=0)
    g_mat += Y_mat * crt_cls_grad

    # build the returns
    grad_b = np.mean(g_mat, axis=1)[:, np.newaxis]
    grad_W = g_mat.dot(X_mat.T) / n_data
    grad_W += lambda_ * W_mat
    return (grad_W, grad_b)
