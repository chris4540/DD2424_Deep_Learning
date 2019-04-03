"""
This file is for the Exercise 1 described in the assignment1.pdf
"""
import pickle
import numpy as np
from simple_network import OneLayerNetwork

def unpickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')

    # convert the key from python2 string to python3 string
    ret = dict()
    for k in data.keys():
        ret[k.decode("utf-8")] = data[k]
    return ret

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

def load_batch(filename):
    """
    Read the batch file and transform the data to double

    Args:
        the file

    Return:
        return a dictionary
    """
    data = unpickle(filename)

    float_pixel = data["data"].T / 255
    onehot_labels = get_label_to_one_hot(data["labels"])
    # build the return
    ret = {
        "pixel_data": float_pixel,
        "onehot_labels": onehot_labels,
        "labels": np.array(data["labels"])
    }

    return ret

if __name__ == '__main__':
    # get the training data
    train_data = load_batch("cifar-10-batches-py/data_batch_1")

    network = OneLayerNetwork(10, 20, lambda_=0.0)
    X_mat = train_data["pixel_data"]
    y_mat = train_data["onehot_labels"]

    # p = network.evaluate(X_mat)

    # cost = network.compute_cost(X_mat, y_mat)
    grad_W, grad_b = network.compute_grad(X_mat[:20, :5], y_mat[:, :5])
    grad_W2, grad_b2 = network.compute_grads_num(X_mat[:20, :5], y_mat[:, :5])

    print("Is grad_W all close:", np.isclose(grad_W, grad_W2))
    print("Is grad_b all close:", np.isclose(grad_b, grad_b2))
