from utils.load_batch import load_batch
from clsr.two_layer_network import TwoLayerNetwork
from time import time


if __name__ == '__main__':
    # load data
    train_data = load_batch("cifar-10-batches-py/data_batch_1")
    valid_data = load_batch("cifar-10-batches-py/data_batch_2")
    test_data = load_batch("cifar-10-batches-py/test_batch")
    ann = TwoLayerNetwork(stop_overfit=False, n_epochs=700, eta=0.05)

    # ann.set_valid_data(valid_data["pixel_data"].T, valid_data["labels"])
    # training time
    st = time()
    ann.fit(train_data["pixel_data"][:, :100].T, train_data["labels"][:100])
    ts = time() - st

    print("Total used time = ", ts)
    score = ann.score(test_data["pixel_data"].T, test_data["labels"])
    print("Accuracy: {}".format(score))
