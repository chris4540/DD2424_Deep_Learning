"""
Weighting initialization subroutines
"""
import numpy as np

def get_random_init(in_dim, out_dim, std, dtype=np.float32):
    """
    This initialization scheme follows assignment1
    """
    W_mat = (std**2) * np.random.randn(out_dim, in_dim).astype(dtype)
    b_vec = (std**2) * np.random.randn(out_dim, 1).astype(dtype)
    return W_mat, b_vec

def get_xavier_init(in_dim, out_dim, dtype=np.float32):
    """

    See also:
    https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94
    """
    W_mat = np.random.randn(out_dim, in_dim).astype(dtype) * np.sqrt(1.0/in_dim)
    b_vec = np.zeros((out_dim, 1), dtype=dtype)
    return W_mat, b_vec
