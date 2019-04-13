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

# def svm_loss_naive(W, X, y, reg):
#     num_classes = W.shape[1]
#     num_train = X.shape[0]
#     loss = 0.0

#     for i in range(num_train):
#         scores = X[i].dot(W)
#         correct_class_score = scores[y[i]]
#         for j in range(num_classes):
#             if j == y[i]:
#                 continue
#             margin = scores[j] - correct_class_score + 1 # note delta = 1
#             if margin > 0:
#                 loss += margin
#     loss /= num_train # mean
#     loss += 0.5 * reg * np.sum(W * W) # l2 regularization
#     return loss

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
    # for non correct classes
    g_mat[g_mat > 1] = 1
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

def compute_grad_fwd_diff(X_mat, Y_mat, W_mat, b_vec, lambda_):
    """
    Translated from matlab version of ComputeGradsNum
    """
    h = 1e-6
    h_inv = 1.0 / h
    nclass = Y_mat.shape[0]
    ndim = X_mat.shape[0]
    grad_W = np.zeros(W_mat.shape)
    grad_b = np.zeros((nclass, 1))

    cost = compute_cost(X_mat, Y_mat, W_mat, b_vec, lambda_)

    for i in range(nclass):
        b_old = b_vec[i, 0]

        b_vec[i, 0] = b_old + h
        new_cost = compute_cost(X_mat, Y_mat, W_mat, b_vec, lambda_)
        grad_b[i, 0] = (new_cost - cost) * h_inv

        b_vec[i, 0] = b_old

    for idx in np.ndindex(W_mat.shape):
        w_old = W_mat[idx]

        W_mat[idx] = w_old + h
        new_cost = compute_cost(X_mat, Y_mat, W_mat, b_vec, lambda_)
        grad_W[idx] = (new_cost - cost) * h_inv

        W_mat[idx] = w_old

    return (grad_W, grad_b)


def compute_grad_central_diff(X_mat, Y_mat, W_mat, b_vec, lambda_):
    """
    Translated from matlab version of ComputeGradsNum
    """
    h = 1e-6
    h_inv = 1.0 / h
    nclass = Y_mat.shape[0]
    ndim = X_mat.shape[0]
    grad_W = np.zeros(W_mat.shape)
    grad_b = np.zeros((nclass, 1))

    for i in range(nclass):
        b_old = b_vec[i, 0]

        b_vec[i, 0] = b_old + h
        c1 = compute_cost(X_mat, Y_mat, W_mat, b_vec, lambda_)

        b_vec[i, 0] = b_old - h
        c2 = compute_cost(X_mat, Y_mat, W_mat, b_vec, lambda_)

        grad_b[i, 0] = (c1 - c2) * h_inv * 0.5

        b_vec[i, 0] = b_old

    for idx in np.ndindex(W_mat.shape):
        w_old = W_mat[idx]

        W_mat[idx] = w_old + h
        c1 = compute_cost(X_mat, Y_mat, W_mat, b_vec, lambda_)

        W_mat[idx] = w_old - h
        c2 = compute_cost(X_mat, Y_mat, W_mat, b_vec, lambda_)

        grad_W[idx] = (c1 - c2) * h_inv * 0.5

        W_mat[idx] = w_old

    return (grad_W, grad_b)
