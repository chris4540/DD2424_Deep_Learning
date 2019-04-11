"""
This file is for the Exercise 1 described in the assignment1.pdf
"""
from one_layer_ann import OneLayerNetwork
from load_batch import load_batch
import visual
from time import time
from load_batch import merge_batch

def get_data():
    datafiles = []
    for i in range(1, 6):
        fname = 'cifar-10-batches-py/data_batch_%d' % i
        datafiles.append(fname)

    merged = merge_batch(datafiles)

    train_data = dict()
    valid_data = dict()
    train_data["pixel_data"] = merged["pixel_data"][:, 1000:]
    train_data["onehot_labels"] = merged["onehot_labels"][:, 1000:]
    train_data["labels"] = merged["labels"][1000:]

    valid_data["pixel_data"] = merged["pixel_data"][:, :1000]
    valid_data["onehot_labels"] = merged["onehot_labels"][:, :1000]
    valid_data["labels"] = merged["labels"][:1000]
    return train_data, valid_data

train_data, valid_data = get_data()
test_data = load_batch("cifar-10-batches-py/test_batch")

def perform_testing(case_tag, lambda_, n_epochs, n_batch, eta):

    nclass = train_data["onehot_labels"].shape[0]
    ndim = train_data["pixel_data"].shape[0]

    # default case
    ann = OneLayerNetwork(nclass, ndim, lambda_)
    ann.set_train_data(train_data["pixel_data"], train_data["onehot_labels"])
    ann.set_valid_data(valid_data["pixel_data"], valid_data["onehot_labels"])
    ann.set_train_params(n_epochs=n_epochs, n_batch=n_batch, eta=eta)
    st = time()
    ann.train()
    ts = time() - st
    print("Total used time = ", ts)
    acc = ann.compute_accuracy(test_data["pixel_data"], test_data["labels"])
    print("{}: Accuracy: {}".format(case_tag, acc))

    # plt = visual.plot_loss(ann)
    # plt.savefig("report/loss_{}.png".format(case_tag))
    # plt.close()

    # plt = visual.plot_weight_mat(ann)
    # plt.savefig("report/wgt_{}.png".format(case_tag))
    # plt.close()

if __name__ == '__main__':
    # perform_testing("case1", lambda_=0.0, n_epochs=40, n_batch=100, eta=0.1)
    perform_testing("case2", lambda_=0.0, n_epochs=40, n_batch=100, eta=0.01)
    # perform_testing("case3", lambda_=0.1, n_epochs=40, n_batch=100, eta=0.01)
    # perform_testing("case4", lambda_=1.0, n_epochs=40, n_batch=100, eta=0.01)
