import numpy as np

def conv_y_to_onehot_mat(labels):
    """
    Create a one hot encoding matrix from labels

    Args:
        labels (list[int])

    Return:
        one hot encoding matrix of the labels
    """
    one_idx = np.array(labels)
    nkind = len(np.unique(one_idx))
    nlabels = len(one_idx)

    ret = np.zeros((nkind, nlabels))
    ret[one_idx, np.arange(nlabels)] = 1
    return ret
