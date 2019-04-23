"""
For reproduce the figure 3 in the assignment
"""
from utils.load_batch import load_batch
from clsr.two_layer_network import TwoLayerNetwork
from time import time
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # load data
    train_data = load_batch("cifar-10-batches-py/data_batch_1")
    valid_data = load_batch("cifar-10-batches-py/data_batch_2")
    test_data = load_batch("cifar-10-batches-py/test_batch")
    params = {
        "stop_overfit": False,
        "n_epochs": 10,
        "lambda_": 0.01,
        "verbose": True,
        "lrate_scheme": {
            "scheme": "cyclic",
            "eta_lim": [1e-5, 1e-1],
            "step_size": 500
        }
    }
    ann = TwoLayerNetwork(**params)

    ann.set_valid_data(valid_data["pixel_data"].T, valid_data["labels"])
    # training time
    st = time()
    ann.fit(train_data["pixel_data"].T, train_data["labels"])
    ts = time() - st

    print("Total used time = ", ts)
    score = ann.score(test_data["pixel_data"].T, test_data["labels"])
    print("Accuracy: {}".format(score))

    # =====================================================================
    iters = range(0, 1001, 100)
    plt.plot(iters, ann.valid_costs, label='validation')
    plt.plot(iters, ann.train_costs, label='training')
    plt.legend(loc='upper right')
    plt.title("Plot training and validation cost at each epoch")
    plt.ylabel("Cost")
    plt.xlabel("Update step")
    plt.ylim(0.0, 4.0)
    plt.xlim(left=0)
    plt.savefig('assign2/cost_plt.png', bbox_inches='tight')
    # =================================================================
    plt.figure()
    plt.plot(iters, ann.train_losses, label='training')
    plt.plot(iters, ann.valid_losses, label='validation')
    plt.legend(loc='upper right')
    plt.title("Plot training and validation loss at each epoch")
    plt.ylabel("Loss")
    plt.xlabel("Update step")
    plt.ylim(0.0, 3)
    plt.xlim(left=0)
    plt.savefig('assign2/loss_plt.png', bbox_inches='tight')

    # =================================================================
    plt.figure()
    plt.plot(iters, ann.train_accuracies, label='training')
    plt.plot(iters, ann.valid_accuracies, label='validation')
    plt.legend(loc='upper right')
    plt.title("Plot training and validation accuracy at each epoch")
    plt.ylabel("accuracy")
    plt.xlabel("Update step")
    plt.ylim(0.0, 1)
    plt.xlim(left=0)
    plt.savefig('assign2/acc_plt.png', bbox_inches='tight')
