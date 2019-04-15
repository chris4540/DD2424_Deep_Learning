"""
Grid search parameters

How to run this code?
-----------------------
python -m scripts.grid_search_params
"""
from utils.load_batch import load_batch
from clsr.one_layer_network import OneLayerNetwork
from time import time
from sklearn.model_selection import ParameterGrid
import json
from time import time

def save_result(results):
    with open("result.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    train_data = load_batch("cifar-10-batches-py/data_batch_1")
    valid_data = load_batch("cifar-10-batches-py/data_batch_2")
    test_data = load_batch("cifar-10-batches-py/test_batch")

    ann = OneLayerNetwork(verbose=True)
    ann.set_valid_data(valid_data["pixel_data"].T, valid_data["labels"])

    param_grid = {
        'lambda_': [0.0, 0.01, 0.05, 0.1],
        'eta': [0.005, 0.01, 0.05],
        'n_batch': [10, 20, 50, 100, 150, 200]
    }
    results = list()

    for i, params in enumerate(ParameterGrid(param_grid)):
        res = dict()

        ann.set_params(**params)
        res['params'] = ann.get_params()

        accs = list()
        train_times = list()
        eval_times = list()
        for _ in range(5):
            start_time = time()
            ann.fit(train_data["pixel_data"].T, train_data["labels"])
            train_times.append(time()-start_time)

            start_time = time()
            acc = ann.score(test_data["pixel_data"].T, test_data["labels"])
            eval_times.append(time()-start_time)

            accs.append(acc)

        res['accs'] = accs
        res['train_times'] = train_times
        res['eval_times'] = eval_times

        results.append(res)
        save_result(results)
