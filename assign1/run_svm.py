"""
Assignment 1
Example to run SupportVectorMachine
"""
from utils.load_batch import load_batch
from clsr.svm import SupportVectorMachine
from time import time

if __name__ == '__main__':
    svm = SupportVectorMachine()

    # load data
    train_data = load_batch("cifar-10-batches-py/data_batch_1")
    valid_data = load_batch("cifar-10-batches-py/data_batch_2")
    test_data = load_batch("cifar-10-batches-py/test_batch")

    svm.set_valid_data(valid_data["pixel_data"].T, valid_data["labels"])
    # training time
    st = time()
    svm.fit(train_data["pixel_data"].T, train_data["labels"])
    ts = time() - st

    print("Total used time = ", ts)
    score = svm.score(test_data["pixel_data"].T, test_data["labels"])
    print("Accuracy: {}".format(score))
