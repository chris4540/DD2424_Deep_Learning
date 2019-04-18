import numpy as np

def compute_grad_fwd_diff(X_mat, Y_mat, W_mat, b_vec, lambda_, cost_fun):
    """
    Translated from matlab version of ComputeGradsNum
    """
    h = 1e-6
    h_inv = 1.0 / h
    nclass = Y_mat.shape[0]
    ndim = X_mat.shape[0]
    grad_W = np.zeros(W_mat.shape)
    grad_b = np.zeros((nclass, 1))

    cost = cost_fun(X_mat, Y_mat, W_mat, b_vec, lambda_)

    for i in range(nclass):
        b_old = b_vec[i, 0]

        b_vec[i, 0] = b_old + h
        new_cost = cost_fun(X_mat, Y_mat, W_mat, b_vec, lambda_)
        grad_b[i, 0] = (new_cost - cost) * h_inv

        b_vec[i, 0] = b_old

    for idx in np.ndindex(W_mat.shape):
        w_old = W_mat[idx]

        W_mat[idx] = w_old + h
        new_cost = cost_fun(X_mat, Y_mat, W_mat, b_vec, lambda_)
        grad_W[idx] = (new_cost - cost) * h_inv

        W_mat[idx] = w_old

    return (grad_W, grad_b)

def compute_grad_central_diff(X_mat, Y_mat, W_mat, b_vec, lambda_, cost_fun):
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
        c1 = cost_fun(X_mat, Y_mat, W_mat, b_vec, lambda_)

        b_vec[i, 0] = b_old - h
        c2 = cost_fun(X_mat, Y_mat, W_mat, b_vec, lambda_)

        grad_b[i, 0] = (c1 - c2) * h_inv * 0.5

        b_vec[i, 0] = b_old

    for idx in np.ndindex(W_mat.shape):
        w_old = W_mat[idx]

        W_mat[idx] = w_old + h
        c1 = cost_fun(X_mat, Y_mat, W_mat, b_vec, lambda_)

        W_mat[idx] = w_old - h
        c2 = cost_fun(X_mat, Y_mat, W_mat, b_vec, lambda_)

        grad_W[idx] = (c1 - c2) * h_inv * 0.5

        W_mat[idx] = w_old

    return (grad_W, grad_b)

def compute_grad_klayers_fwd_diff(X_mat, Y_mat, W_mats, b_vecs, lambda_, cost_fun):
    """
    """
    h = 1e-6
    h_inv = 1.0 / h

    cost = cost_fun(X_mat, Y_mat, W_mats, b_vecs, lambda_)

    grad_Ws = list()
    grad_bs = list()
    for W_mat, b_vec in zip(W_mats, b_vecs):

        # -------------
        # grad_b
        # -------------
        grad_b = np.zeros(b_vec.shape)
        for idx in np.ndindex(b_vec.shape):
            b_old = b_vec[idx]

            b_vec[idx] = b_old + h
            new_cost = cost_fun(X_mat, Y_mat, W_mats, b_vecs, lambda_)
            grad_b[idx] = (new_cost - cost) * h_inv
            # put back the old value
            b_vec[idx] = b_old

        grad_bs.append(grad_b)
        # -------------
        # grad_W
        # -------------
        grad_W = np.zeros(W_mat.shape)
        for idx in np.ndindex(W_mat.shape):
            w_old = W_mat[idx]

            W_mat[idx] = w_old + h
            new_cost = cost_fun(X_mat, Y_mat, W_mats, b_vecs, lambda_)
            grad_W[idx] = (new_cost - cost) * h_inv
            W_mat[idx] = w_old

        grad_Ws.append(grad_W)

    return (grad_Ws, grad_bs)

def compute_grad_klayers_cent_diff(X_mat, Y_mat, W_mats, b_vecs, lambda_, cost_fun):
    """
    """
    h = 1e-6
    h_inv = 1.0 / h

    grad_Ws = list()
    grad_bs = list()
    for W_mat, b_vec in zip(W_mats, b_vecs):

        # -------------
        # grad_b
        # -------------
        grad_b = np.zeros(b_vec.shape)
        for idx in np.ndindex(b_vec.shape):
            b_old = b_vec[idx]

            b_vec[idx] = b_old + h
            c1 = cost_fun(X_mat, Y_mat, W_mats, b_vecs, lambda_)
            b_vec[idx] = b_old - h
            c2 = cost_fun(X_mat, Y_mat, W_mats, b_vecs, lambda_)
            grad_b[idx] = (c1 - c2) * h_inv * 0.5
            # put back the old value
            b_vec[idx] = b_old

        grad_bs.append(grad_b)
        # -------------
        # grad_W
        # -------------
        grad_W = np.zeros(W_mat.shape)
        for idx in np.ndindex(W_mat.shape):
            w_old = W_mat[idx]

            W_mat[idx] = w_old + h
            c1 = cost_fun(X_mat, Y_mat, W_mats, b_vecs, lambda_)

            W_mat[idx] = w_old - h
            c2 = cost_fun(X_mat, Y_mat, W_mats, b_vecs, lambda_)

            grad_W[idx] = (c1 - c2) * h_inv * 0.5
            W_mat[idx] = w_old

        grad_Ws.append(grad_W)

    return (grad_Ws, grad_bs)
