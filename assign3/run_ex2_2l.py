"""
"""
from utils.load_batch import load_batch
from clsr.k_layer_network import KLayerNetwork
from time import time

if __name__ == '__main__':
    # load data
    train_data = load_batch("cifar-10-batches-py/data_batch_1")
    valid_data = load_batch("cifar-10-batches-py/data_batch_2")
    test_data = load_batch("cifar-10-batches-py/test_batch")
    params = {
        "stop_overfit": False,
        "n_epochs": 500,
        "lambda_": 0,
        "verbose": True,
        "lrate_scheme": {
            "scheme": "cyclic",
            "eta_lim": [1e-5, 1e-1],
            "step_size": 500
        }
    }
    ann = KLayerNetwork(n_layers=4, n_hidden_nodes=[50, 50, 50], **params)

    # ann.set_valid_data(valid_data["pixel_data"].T, valid_data["labels"])
    # training time
    st = time()
    ann.fit(train_data["pixel_data"][:, :100].T, train_data["labels"][:100])
    ts = time() - st

    print("Total used time = ", ts)
    score = ann.score(test_data["pixel_data"].T, test_data["labels"])
    print("Accuracy: {}".format(score))
