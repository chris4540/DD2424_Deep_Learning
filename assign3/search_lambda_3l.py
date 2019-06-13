"""
Assignment 3 3-layer
"""
import numpy as np
from utils.load_batch import load_batch
from utils.load_batch import cifar10_DataLoader
from utils.handle_data import data_split
from utils.handle_data import get_all_train_data
from utils.preprocess import StandardScaler
from utils.lrate import CyclicLR
from clsr.nn_kl import KLayerNeuralNetwork
import time
from utils import train
from utils import evaluate
import json



def draw_lambda_val(l_min, l_max):
    """
    Args:
        l_min (float): the min of lambda in log scale
        l_max (float): the max of lambda in log scale
    """
    log_lambda = np.random.uniform(low=l_min, high=l_max)
    return 10**log_lambda


def get_valid_score(train_loader, valid_loader, n_epoch, step_size, weight_decay):
    print("weight_decay: ", weight_decay)
    net = KLayerNeuralNetwork(n_hidden_nodes=[50, 50], p_dropout=0.0, batch_norm=True, batch_norm_momentum=0.7)
    scheduler = CyclicLR(eta_min=1e-5, eta_max=1e-1, step_size=step_size)
    # =========================================================================
    best_valid_acc = -np.inf
    for epoch in range(n_epoch):
        train(train_loader, net, weight_decay, scheduler)
        valid_acc = evaluate(valid_loader, net)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc

        print("[Evaluate] Valid. Acc.: %.3f%%" % (
            valid_acc*100))

    print("[Result] Valid. Acc.: %.3f%%" % (
        best_valid_acc*100))

    return best_valid_acc

def save_result(results, json_name):
    with open(json_name, 'w') as f:
        json.dump(results, f, indent=2)



if __name__ == "__main__":
    merged_data = get_all_train_data("cifar-10-batches-py")
    train_data, valid_data = data_split(merged_data, n_valid=5000)
    test_data = load_batch("cifar-10-batches-py/test_batch")

    scaler = StandardScaler()
    scaler.fit(train_data['pixel_data'])
    train_data['pixel_data'] = scaler.transform(train_data['pixel_data'])
    valid_data['pixel_data'] = scaler.transform(valid_data['pixel_data'])
    test_data['pixel_data'] = scaler.transform(test_data['pixel_data'])
    print("Done preprocessing!")
    # ==================================================================
    # make dataloader
    batch_size = 100
    train_loader = cifar10_DataLoader(train_data, batch_size=batch_size)
    valid_loader = cifar10_DataLoader(valid_data, batch_size=batch_size)
    test_loader = cifar10_DataLoader(test_data, batch_size=batch_size)
    # ==================================================================
    ntrain = train_data['labels'].shape[0]
    n_step_per_cycle = 5
    ncycle = 2
    n_epoch = ncycle*n_step_per_cycle*2
    iter_per_epoch = int(np.ceil(ntrain / batch_size))
    step_size = n_step_per_cycle*iter_per_epoch
    print("--------- Train Schedule ---------")
    print("ncycle: ", ncycle)
    print("n_epoch: ", n_epoch)
    print("step_size: ", step_size)
    print("iter_per_epoch: ", iter_per_epoch)
    print("n_step_per_cycle: ", n_step_per_cycle)
    print("--------- Train Schedule ---------")

    results = list()
    for _ in range(20):
        w_decay = draw_lambda_val(-5, -3)
        best_valid = get_valid_score(train_loader, valid_loader, step_size, w_decay)
        results.append(
            {
                "weight_decay": w_decay,
                "best_valid": best_valid,
            }
        )
        save_result(results, 'tmp.json')
