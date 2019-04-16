"""
A script to run OneLayerNetwork
"""
from utils.load_batch import load_batch
from utils.load_batch import merge_batch
from clsr.one_layer_network import OneLayerNetwork
from time import time
from utils import visual

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

if __name__ == '__main__':
    # load data
    # train_data = load_batch("cifar-10-batches-py/data_batch_1")
    # valid_data = load_batch("cifar-10-batches-py/data_batch_2")
    train_data, valid_data = get_data()
    test_data = load_batch("cifar-10-batches-py/test_batch")


    # ann = OneLayerNetwork(decay=0.95, shuffle_per_epoch=False, n_epochs=200)
    # ann = OneLayerNetwork(decay=1.0, eta=0.005, n_epochs=200)
    params = {'eta': 0.02, 'verbose': True, 'lambda_': 0.0, 'decay': 0.9, 'n_batch': 20, "n_epochs": 500, "shuffle_per_epoch": True}
    #params = {'lambda_': 0.0, 'eta': 0.005, 'decay': 1.0, 'n_batch': 50}
    ann = OneLayerNetwork(**params)
    ann.set_valid_data(valid_data["pixel_data"].T, valid_data["labels"])
    # training time
    st = time()
    ann.fit(train_data["pixel_data"].T, train_data["labels"])
    ts = time() - st

    print("Total used time = ", ts)
    score = ann.score(test_data["pixel_data"].T, test_data["labels"])
    print("Accuracy: {}".format(score))
    plt = visual.plot_loss(ann)
    plt.savefig("shuffle_per_epoch.png")
    plt.close()
