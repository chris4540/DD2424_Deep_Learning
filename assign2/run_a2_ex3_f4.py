"""
For reproduce the figure 4 in the assignment
"""
from utils.load_batch import load_batch
from clsr.two_layer_network import TwoLayerNetwork
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # load data
    train_data = load_batch("cifar-10-batches-py/data_batch_1")
    valid_data = load_batch("cifar-10-batches-py/data_batch_2")
    test_data = load_batch("cifar-10-batches-py/test_batch")
    # =================================================
    n_s = 800
    n_cycle = 3
    n_iters = 2*n_s*n_cycle
    n_batch = 100
    n_train_data = len(train_data['labels'])
    n_epochs = int(n_iters*n_batch/n_train_data)
    params = {
        "stop_overfit": False,
        "n_epochs": n_epochs,
        "n_batch": n_batch,
        "lambda_": 0.01,
        "verbose": True,
        "lrate_scheme": {
            "scheme": "cyclic",
            "eta_lim": [1e-5, 1e-1],
            "step_size": n_s
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
    iters = ann.iters
    plt.plot(iters, ann.valid_costs, label='validation')
    plt.plot(iters, ann.train_costs, label='training')
    plt.legend(loc='upper right')
    plt.title("Plot training and validation cost at each epoch")
    plt.ylabel("Cost")
    plt.xlabel("Update step")
    plt.ylim(0.0, 4.0)
    plt.xlim(left=0)
    plt.savefig('assign2/f4_cost_plt.png', bbox_inches='tight')
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
    plt.savefig('assign2/f4_loss_plt.png', bbox_inches='tight')

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
    plt.savefig('assign2/f4_acc_plt.png', bbox_inches='tight')
