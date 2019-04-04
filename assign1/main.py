"""
This file is for the Exercise 1 described in the assignment1.pdf
"""
from one_layer_ann import OneLayerNetwork
from load_batch import load_batch

if __name__ == '__main__':
    # get the training data
    train_data = load_batch("cifar-10-batches-py/data_batch_1")

    X_mat = train_data["pixel_data"]
    Y_mat = train_data["onehot_labels"]

    nclass = Y_mat.shape[0]
    ndim = X_mat.shape[0]
    ann = OneLayerNetwork(nclass, ndim)

    ann.set_train_data(X_mat, Y_mat)
    ann.train()
