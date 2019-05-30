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
        print("Lambda: {};Accuracy: {}".format(lambda_, score))
        scores.append(score)

    # ===================================
    ret = dict()
    ret['lambda_'] = lambda_
    ret['scores'] = scores
    ret['mean_score'] = np.mean(scores)
    return ret

def save_result(results, json_name):
    with open(json_name, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    # settings
    n_search = 20
    search_type = "fine" # or "coarse"

    if search_type == "coarse":
        l_min, l_max = -5, -1
        n_cycle = 2
        k = 2
    elif search_type == "fine":
        l_min, l_max = -3, -2
        n_cycle = 3
        k = 3
    # ===============================================================
    # READ DATA
    merged_data = get_all_train_data("cifar-10-batches-py")
    train_data, valid_data = data_split(merged_data, n_valid=5000)
    # ===============================================================
    n_train_data = len(train_data['labels'])
    n_batch = 100
    n_s = int(k*np.floor(n_train_data/n_batch))
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


    result_json = "a2_ex4_%s_search.json" % search_type

    results = list()
    for _ in range(n_search):
        lambda_ = draw_lambda_val(l_min, l_max)
        ret = get_performance(train_data, valid_data, params, lambda_)
        results.append(ret)
        # dump a json whenever one search finish, for resuming the search process
        save_result(results, result_json)

