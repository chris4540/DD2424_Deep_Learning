from one_layer_ann import OneLayerNetwork
from load_batch import load_batch
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

    nclass = train_data["onehot_labels"].shape[0]
    ndim = train_data["pixel_data"].shape[0]

    ann = OneLayerNetwork(verbose=False)
    ann.set_train_data(train_data["pixel_data"], train_data["onehot_labels"])
    ann.set_valid_data(valid_data["pixel_data"], valid_data["onehot_labels"])

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
            ann.train()
            train_times.append(time()-start_time)

            start_time = time()
            acc = ann.compute_accuracy(test_data["pixel_data"], test_data["labels"])
            eval_times.append(time()-start_time)

            print(acc)
            accs.append(acc)

        res['accs'] = accs
        res['train_times'] = train_times
        res['eval_times'] = eval_times

        results.append(res)
        save_result(results)
