import numpy as np

def compute_cost(X_mat, Y_mat, W_mat, b_vec, lambda_):
    """
    \sum_{i!=y}[max(0, \vec{s} - s_y + 1)]
    """
    n_data = X_mat.shape[1]
    s_mat = W_mat.dot(X_mat) + b_vec
    s_y_vec = np.sum(s_mat * Y_mat, axis=0) # shape = (ndata,)
    hinge_mat = np.maximum(s_mat - s_y_vec + 1, 0)
    ret = (np.sum(hinge_mat) - n_data) / n_data + 0.5*lambda_*np.sum(W_mat**2)

    return ret
