"""
{'lambda_': 0.0, 'eta': 0.005, 'decay': 1.0, 'n_batch': 50, 'verbose': False}
"""
import pickle
import numpy as np
from utils.load_batch import merge_batch
from utils.load_batch import load_batch
from clsr.one_layer_network import OneLayerNetwork
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    parameters = {
        'lambda_': list(np.linspace(0.0, 1.0, 9)) + [1.5, 2.0],
        'eta': [0.005, 0.01, 0.02, 0.05, 0.1],
        'n_batch': [10, 20, 50, 100, 150, 200],
        'verbose': [False],
        'decay': [1.0],
    }

    # load data
    train_data = merge_batch(
        ["cifar-10-batches-py/data_batch_1",
         "cifar-10-batches-py/data_batch_2"
        ])
    test_data = load_batch("cifar-10-batches-py/test_batch")

    ann = OneLayerNetwork()
    clf = GridSearchCV(ann, parameters, cv=2, n_jobs=8)
    clf.fit(train_data["pixel_data"].T, train_data["labels"])

    print(clf.best_params_)
    best_est = clf.best_estimator_
    print(clf.best_score_)
    with open('GridSearchCV.pkl', 'wb') as pkl:
        pickle.dump(clf, pkl, pickle.HIGHEST_PROTOCOL)
