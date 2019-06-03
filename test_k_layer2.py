"""
-------- TRAINING PARAMS --------
dtype: float32
verbose: True
wgt_init: xavier
p_dropout: 0.0
n_features: 3072
n_classes: 10
n_hidden_nodes: [750, 400, 200, 100, 50, 30, 20]
batch_norm: True
batch_norm_momentum: 0.7
-------- TRAINING PARAMS --------
Weightings and bias are initialized with xavier method.
--------- Train Schedule ---------
ncycle:  2
n_epoch:  20
step_size:  2250
iter_per_epoch:  450
n_step_per_cycle:  5
weight_decay:  0.005
--------- Train Schedule ---------
Train Time used: 50      Loss: 1.002 | Train Acc: 66.347% (29856/45000)
[Evaluate] Valid. Acc.: 55.440%          Test Acc.: 55.420%
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
    # net = KLayerNeuralNetwork(n_hidden_nodes=[50, 50], p_dropout=0.0)
    # net = KLayerNeuralNetwork(n_hidden_nodes=[70, 50], p_dropout=0.0)
    # net = KLayerNeuralNetwork(p_dropout=0.0,
    #     n_hidden_nodes= [50, 30, 20, 20, 10, 10, 10, 10], batch_norm=False
    #     )
    # net = KLayerNeuralNetwork(p_dropout=0.0,
    #     n_hidden_nodes= [50, 30, 20, 20, 10, 10, 10, 10], batch_norm=True
    #     )
    net = KLayerNeuralNetwork(p_dropout=0.2,
        n_hidden_nodes= [500, 250, 100, 50], batch_norm=True,
        batch_norm_momentum=0.9)
    ntrain = train_data['labels'].shape[0]
    n_step_per_cycle = 5
    ncycle = 10
    n_epoch = ncycle*n_step_per_cycle*2
    # n_epoch = 1
    iter_per_epoch = int(np.ceil(ntrain / batch_size))
    step_size = n_step_per_cycle*iter_per_epoch
    # weight_decay = 0.005
    weight_decay = 0.00
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
        # train(valid_loader, net, weight_decay, scheduler)
        valid_acc = evaluate(valid_loader, net)
        test_acc = evaluate(test_loader, net)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            corrs_test_acc = test_acc

        print("[Evaluate] Valid. Acc.: %.3f%% \t Test Acc.: %.3f%%" % (
            valid_acc*100, test_acc*100))

    print("[Result] Valid. Acc.: %.3f%% \t Test Acc.: %.3f%%" % (
        best_valid_acc*100, corrs_test_acc*100))
