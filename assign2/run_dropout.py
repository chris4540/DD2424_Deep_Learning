"""
Assignment 2 bonus point quesiton:

Basic part: for checking sanity
    n_hidden = 50
    weight_decay = 0.0
    k = 3; cycle = 3
------------------------------------------------------------------
    p_dropout = 0.0
    [Result] Valid. Acc.: 53.500%    Test Acc.: 49.960%
------------------------------------------------------------------
    p_dropout = 0.01
    [Result] Valid. Acc.: 53.500%    Test Acc.: 51.750%
------------------------------------------------------------------
    p_dropout = 0.8
    [Result] Valid. Acc.: 40.500%    Test Acc.: 39.390%
================================================================================
Advacne part:
    n_hidden = 800
    weight_decay = 0.0
    k = 3; cycle = 3
------------------------------------------------------------------
    p_dropout = 0.0
------------------------------------------------------------------
    p_dropout = 0.1
    [Result] Valid. Acc.: 57.800%    Test Acc.: 56.430%
------------------------------------------------------------------
    p_dropout = 0.3
------------------------------------------------------------------
    p_dropout = 0.5
"""
import numpy as np
from utils.load_batch import load_batch
from utils.load_batch import cifar10_DataLoader
from utils.handle_data import data_split
from utils.handle_data import get_all_train_data
from utils.preprocess import StandardScaler
from utils.lrate import CyclicLR
from clsr.nn_2l import TwoLayerNeuralNetwork
import time
from utils import train
from utils import evaluate

if __name__ == "__main__":
    merged_data = get_all_train_data("cifar-10-batches-py")
    train_data, valid_data = data_split(merged_data, n_valid=1000)
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
    net = TwoLayerNeuralNetwork(n_hidden_nodes=[800], p_dropout=0.3)
    # net = TwoLayerNeuralNetwork(n_hidden_nodes=[768])
    ntrain = train_data['labels'].shape[0]
    n_step_per_cycle = 3
    ncycle = 3
    n_epoch = ncycle*n_step_per_cycle*2
    iter_per_epoch = int(np.ceil(ntrain / batch_size))
    step_size = n_step_per_cycle*iter_per_epoch
    # weight_decay = 0.004454
    # weight_decay = 0.005
    weight_decay = 0.000
    scheduler = CyclicLR(eta_min=1e-5, eta_max=1e-1, step_size=step_size)
    # =========================================================================
    print("--------- Train Schedule ---------")
    print("ncycle: ", ncycle)
    print("n_epoch: ", n_epoch)
    print("step_size: ", step_size)
    print("iter_per_epoch: ", iter_per_epoch)
    print("n_step_per_cycle: ", n_step_per_cycle)
    print("weight_decay: ", weight_decay)
    print("--------- Train Schedule ---------")
    best_valid_acc = -np.inf
    for epoch in range(n_epoch):
        train(train_loader, net, weight_decay, scheduler)
        valid_acc = evaluate(valid_loader, net)
        test_acc = evaluate(test_loader, net)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            corrs_test_acc = test_acc

        print("[Evaluate] Valid. Acc.: %.3f%% \t Test Acc.: %.3f%%" % (
            valid_acc*100, test_acc*100))

    print("[Result] Valid. Acc.: %.3f%% \t Test Acc.: %.3f%%" % (
        best_valid_acc*100, corrs_test_acc*100))