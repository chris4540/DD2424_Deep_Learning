"""
For the question v
"""
import numpy as np
import matplotlib.pyplot as plt
from utils.handle_data import get_all_train_data
from clsr.two_layer_network import TwoLayerNetwork
from utils.load_batch import load_batch
from utils.handle_data import data_split


if __name__ == "__main__":
    # settings
    n_cycle = 3 # will overshot when too much cycles
    k = 3
    lambda_ = 0.000142
    # ===============================================================
    # READ DATA
    merged_data = get_all_train_data("cifar-10-batches-py")
    train_data, valid_data = data_split(merged_data, n_valid=1000)
    test_data = load_batch("cifar-10-batches-py/test_batch")
    # ===============================================================
    n_train_data = len(merged_data['labels'])
    n_batch = 100
    n_s = int(k*np.floor(n_train_data/n_batch))
    n_iters = 2*n_s*n_cycle
    n_epochs = int(n_iters*n_batch/n_train_data)
    # ======================================================
    params = {
        "n_epochs": n_epochs,
        "n_batch": n_batch,
        "verbose": True,
        "lambda_": lambda_,
        "record_training": True,
        "lrate_scheme": {
            "scheme": "cyclic",
            "eta_lim": [1e-5, 1e-1],
            "step_size": n_s
        }
    }
    clsr = TwoLayerNetwork(**params)
    clsr.set_valid_data(valid_data["pixel_data"].T, valid_data["labels"])
    clsr.fit(train_data["pixel_data"].T, train_data["labels"])
    # ==============================================================
    valid_score = clsr.score(valid_data["pixel_data"].T, valid_data["labels"])
    test_score = clsr.score(test_data["pixel_data"].T, test_data["labels"])
    print("Lambda: {};valid acc.:{} test acc.: {}".format(lambda_, valid_score, test_score))

    # =====================================================================
    iters = clsr.iters
    plt.plot(iters, clsr.valid_costs, label='validation')
    plt.plot(iters, clsr.train_costs, label='training')
    plt.legend(loc='upper right')
    plt.ylabel("Cost")
    plt.xlabel("Update step")
    plt.ylim(0.0, 4.0)
    plt.xlim(left=0)
    plt.savefig('assign2/f5_cost_plt.png', bbox_inches='tight')
    # =================================================================
    plt.figure()
    plt.plot(iters, clsr.train_losses, label='training')
    plt.plot(iters, clsr.valid_losses, label='validation')
    plt.legend(loc='upper right')
    plt.ylabel("Loss")
    plt.xlabel("Update step")
    plt.ylim(0.0, 3)
    plt.xlim(left=0)
    plt.savefig('assign2/f5_loss_plt.png', bbox_inches='tight')

    # =================================================================
    plt.figure()
    plt.plot(iters, clsr.train_accuracies, label='training')
    plt.plot(iters, clsr.valid_accuracies, label='validation')
    plt.legend(loc='upper right')
    plt.ylabel("accuracy")
    plt.xlabel("Update step")
    plt.ylim(0.0, 1)
    plt.xlim(left=0)
    plt.savefig('assign2/f5_acc_plt.png', bbox_inches='tight')

