import numpy as np
from utils.handle_data import get_all_train_data
from clsr.two_layer_network import TwoLayerNetwork
from utils.load_batch import load_batch
from utils.handle_data import data_split


if __name__ == "__main__":
    # settings
    n_search = 20
    n_cycle = 3 # will overshot when too much cycles
    k = 2
    log_lambda = -2.844722
    # log_lambda = -3.840433
    lambda_ = 10**log_lambda
    # ===============================================================
    # READ DATA
    merged_data = get_all_train_data("cifar-10-batches-py")
    train_data, valid_data = data_split(merged_data, n_valid=1000)
    test_data = load_batch("cifar-10-batches-py/test_batch")
    # ===============================================================
    n_train_data = len(merged_data['labels'])
    n_batch = 100
    n_s = int(k*np.floor(n_train_data/n_batch))
    n_iters = 2*n_s*n_cycle
    n_epochs = int(n_iters*n_batch/n_train_data)
    # ======================================================
    params = {
        "n_epochs": n_epochs,
        "n_batch": n_batch,
        "verbose": True,
        "lambda_": lambda_,
        "record_training": False,
        "lrate_scheme": {
            "scheme": "cyclic",
            "eta_lim": [1e-5, 1e-1],
            "step_size": n_s
        }
    }
    clsr = TwoLayerNetwork(**params)
    clsr.set_valid_data(valid_data["pixel_data"].T, valid_data["labels"])
    clsr.fit(train_data["pixel_data"].T, train_data["labels"])
    # ==============================================================
    valid_score = clsr.score(valid_data["pixel_data"].T, valid_data["labels"])
    test_score = clsr.score(test_data["pixel_data"].T, test_data["labels"])
    print("Lambda: {};valid acc.:{} test acc.: {}".format(lambda_, valid_score, test_score))


