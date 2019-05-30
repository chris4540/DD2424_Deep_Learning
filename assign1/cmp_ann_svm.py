"""

"""
from utils.load_batch import load_batch
from clsr.one_layer_network import OneLayerNetwork
from clsr.svm import SupportVectorMachine

if __name__ == '__main__':
    # load data
    train_data = load_batch("cifar-10-batches-py/data_batch_1")
    valid_data = load_batch("cifar-10-batches-py/data_batch_2")
    test_data = load_batch("cifar-10-batches-py/test_batch")

    params = {
        'eta': 0.02,
        'verbose': True,
        'lambda_': 0.0,
        'decay': 0.9,
        'n_batch': 20,
        'n_epochs': 40,
    }
    # params['n_batch'] = 50
    # params['eta'] = 0.005
    # params['decay'] = 1.0
    # params['lambda_'] = 0.05
    params['lambda_'] = 0.1


    # SVM
    svm = SupportVectorMachine(**params)
    # training time
    svm.set_valid_data(valid_data["pixel_data"].T, valid_data["labels"])
    svm.fit(train_data["pixel_data"].T, train_data["labels"])

    score = svm.score(test_data["pixel_data"].T, test_data["labels"])
    print("SVM Accuracy: {}".format(score))

    # ANN
    ann = OneLayerNetwork(**params)
    # training time
    ann.set_valid_data(valid_data["pixel_data"].T, valid_data["labels"])
    ann.fit(train_data["pixel_data"].T, train_data["labels"])

    score = ann.score(test_data["pixel_data"].T, test_data["labels"])
    print("ANN Accuracy: {}".format(score))
