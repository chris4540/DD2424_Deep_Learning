"""
Module for obtaining and handling data. It is refactored from old code
"""
import numpy as np
from os.path import join
from .load_batch import merge_batch

def get_all_train_data(cifar_dir):
    """
    Obtain all training data
    """
    datafiles = []
    data_fname_pattern = join(cifar_dir, "data_batch_%d")
    for i in range(1, 6):
        fname = data_fname_pattern % i
        datafiles.append(fname)

    merged = merge_batch(datafiles)
    return merged

def data_split(merged_data, n_valid=1000):
    """
    Spite data to training set and validation set
    """
    train_data = dict()
    valid_data = dict()

    # 2d data
    for k in ["pixel_data"]:
        train_data[k] = merged_data[k][:, n_valid:]
        valid_data[k] = merged_data[k][:, :n_valid]
    # 1d data
    for k in ["labels"]:
        train_data[k] = merged_data[k][n_valid:]
        valid_data[k] = merged_data[k][:n_valid]

    return train_data, valid_data

def get_label_to_one_hot(labels):
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