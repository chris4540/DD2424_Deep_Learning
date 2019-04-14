"""
Example to run SupportVectorMachine
"""
from utils.load_batch import load_batch
from clsr.one_layer_network import OneLayerNetwork
from time import time
from utils import visual

if __name__ == '__main__':
    # load data
    train_data = load_batch("cifar-10-batches-py/data_batch_1")
    valid_data = load_batch("cifar-10-batches-py/data_batch_2")
    test_data = load_batch("cifar-10-batches-py/test_batch")


    ann = OneLayerNetwork(decay=1, shuffle_per_epoch=True)
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
