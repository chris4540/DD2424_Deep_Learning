"""
"""
import numpy as np
from time import time
from clsr.k_layer_network import KLayerNetwork
from utils.load_batch import load_batch
from utils.handle_data import get_all_train_data
from utils.handle_data import data_split

if __name__ == '__main__':
    # settings
    n_cycle = 3 # will overshot when too much cycles
    k = 5
    lambda_ = 0.005
    eta_min = 1e-5
    eta_max = 1e-1
    n_batch = 100
    # ===============================================================
    # READ DATA
    merged_data = get_all_train_data("cifar-10-batches-py")
    train_data, valid_data = data_split(merged_data, n_valid=5000)
    test_data = load_batch("cifar-10-batches-py/test_batch")
    # ===============================================================
    n_train_data = len(train_data['labels'])
    n_s = int(k*np.floor(n_train_data/n_batch))
    n_iters = 2*n_s*n_cycle
    n_epochs = int(n_iters*n_batch/n_train_data)
    # ======================================================
    params = {
        "stop_overfit": False,
        "n_epochs": n_epochs,
        "lambda_": lambda_,
        "verbose": True,
        "lrate_scheme": {
            "scheme": "cyclic",
            "eta_lim": [eta_min, eta_max],
            "step_size": n_s
        },
        "record_training": False
    }
    clsr = KLayerNetwork(n_hidden_nodes=[50, 50], **params)
    clsr.set_valid_data(valid_data["pixel_data"].T, valid_data["labels"])
    # training time
    st = time()
    clsr.fit(train_data["pixel_data"].T, train_data["labels"])
    ts = time() - st

    print("Total used time = ", ts)
    score = clsr.score(test_data["pixel_data"].T, test_data["labels"])
    print("Accuracy: {}".format(score))
