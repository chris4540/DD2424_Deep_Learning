"""
This file is for the Exercise 1 described in the assignment1.pdf

Run cmd:
python -m scripts.plt_ex1_basic
"""
from utils.load_batch import load_batch
from clsr.one_layer_network import OneLayerNetwork
from time import time
from utils import visual

# get the training data
train_data = load_batch("cifar-10-batches-py/data_batch_1")
valid_data = load_batch("cifar-10-batches-py/data_batch_2")
test_data = load_batch("cifar-10-batches-py/test_batch")

def perform_testing(case_tag, lambda_, n_epochs, n_batch, eta):

    # default case
    ann = OneLayerNetwork(
        decay=1.0, eta=eta, n_epochs=n_epochs, lambda_=lambda_, n_batch=n_batch,
        stop_overfit=False)
    ann.set_valid_data(valid_data["pixel_data"].T, valid_data["labels"])

    # training time
    st = time()
    ann.fit(train_data["pixel_data"].T, train_data["labels"])
    ts = time() - st

    print("Total used time = ", ts)
    score = ann.score(test_data["pixel_data"].T, test_data["labels"])
    print("Accuracy: {}".format(score))
    print("===========================================================")
    plt = visual.plot_loss(ann)
    plt.savefig("report/loss_{}.png".format(case_tag))
    plt.close()

    plt = visual.plot_weight_mat(ann)
    plt.savefig("report/wgt_{}.png".format(case_tag))
    plt.close()

if __name__ == '__main__':
    perform_testing("case1", lambda_=0.0, n_epochs=40, n_batch=100, eta=0.1)
    perform_testing("case2", lambda_=0.0, n_epochs=40, n_batch=100, eta=0.01)
    perform_testing("case3", lambda_=0.1, n_epochs=40, n_batch=100, eta=0.01)
    perform_testing("case4", lambda_=1.0, n_epochs=40, n_batch=100, eta=0.01)
