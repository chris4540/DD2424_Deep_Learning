import numpy as np
from utils.handle_data import get_all_train_data
from utils.handle_data import data_split
from clsr.two_layer_network import TwoLayerNetwork
import json

def draw_lambda_val(l_min, l_max):
    """
    Args:
        l_min (float): the min of lambda in log scale
        l_max (float): the max of lambda in log scale
    """
    log_lambda = np.random.uniform(low=l_min, high=l_max)
    return 10**log_lambda

def get_performance(train_data, valid_data, params, lambda_, n_try=3):
    params['lambda_'] = lambda_
    # ========================================
    scores = list()
    clsr = TwoLayerNetwork(**params)

    for _ in range(n_try):
        clsr.fit(train_data["pixel_data"].T, train_data["labels"])
        score = clsr.score(valid_data["pixel_data"].T, valid_data["labels"])
        print("Accuracy: {}".format(score))
        scores.append(score)

    # ===================================
    ret = dict()
    ret['lambda_'] = lambda_
    ret['scores'] = scores
    ret['mean_score'] = np.mean(scores)
    return ret

def save_result(results):
    with open("a2_ex4_coarse_search.json", 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    n_search = 20
    # ===============================================================
    # READ DATA
    merged_data = get_all_train_data("cifar-10-batches-py")
    train_data, valid_data = data_split(merged_data, n_valid=5000)
    # ===============================================================
    n_train_data = len(train_data['labels'])
    n_batch = 100
    n_s = int(2*np.floor(n_train_data/n_batch))

    n_cycle = 2
    n_iters = 2*n_s*n_cycle
    n_epochs = int(n_iters*n_batch/n_train_data)
    # ======================================================
    params = {
        "stop_overfit": False,
        "n_epochs": n_epochs,
        "n_batch": n_batch,
        "verbose": True,
        "record_training": False,
        "lrate_scheme": {
            "scheme": "cyclic",
            "eta_lim": [1e-5, 1e-1],
            "step_size": n_s
        }
    }

    results = list()
    for _ in range(n_search):
        lambda_ = draw_lambda_val(-5, -1)
        ret = get_performance(train_data, valid_data, params, lambda_)
        results.append(ret)
        # dump a json whenever one search finish, for resuming the search process
        save_result(results)
