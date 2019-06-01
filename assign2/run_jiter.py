"""
Assignment 2 bonus point quesiton:
>> Augment your training data by applying a small random geometric and photometric jitter
-------- TRAINING PARAMS --------
dtype: float32
verbose: True
wgt_init: xavier
p_dropout: 0.0
n_features: 3072
n_classes: 10
n_hidden_nodes: [50]
-------- TRAINING PARAMS --------
Weightings and bias are initialized with xavier method.
--------- Train Schedule ---------
ncycle:  2
n_epoch:  8
step_size:  980
iter_per_epoch:  490
n_step_per_cycle:  2
weight_decay:  0.0
--------- Train Schedule ---------

with jiter:
Train Time used: 32      Loss: 1.879 | Train Acc: 33.337% (16335/49000)
[Evaluate] Valid. Acc.: 39.400%          Test Acc.: 41.990%
Train Time used: 32      Loss: 1.657 | Train Acc: 41.851% (20507/49000)
[Evaluate] Valid. Acc.: 41.000%          Test Acc.: 42.930%
Train Time used: 32      Loss: 1.550 | Train Acc: 45.533% (22311/49000)
[Evaluate] Valid. Acc.: 48.200%          Test Acc.: 48.970%
Train Time used: 32      Loss: 1.419 | Train Acc: 50.404% (24698/49000)
[Evaluate] Valid. Acc.: 50.300%          Test Acc.: 51.200%
Train Time used: 32      Loss: 1.396 | Train Acc: 51.276% (25125/49000)
[Evaluate] Valid. Acc.: 48.400%          Test Acc.: 49.420%
Train Time used: 31      Loss: 1.488 | Train Acc: 47.959% (23500/49000)
[Evaluate] Valid. Acc.: 45.600%          Test Acc.: 45.380%
Train Time used: 32      Loss: 1.462 | Train Acc: 48.971% (23996/49000)
[Evaluate] Valid. Acc.: 49.300%          Test Acc.: 50.660%
Train Time used: 32      Loss: 1.346 | Train Acc: 53.202% (26069/49000)
[Evaluate] Valid. Acc.: 52.400%          Test Acc.: 52.060%
[Result] Valid. Acc.: 52.400%    Test Acc.: 52.060%
================================================================================
Train Time used: 4       Loss: 1.840 | Train Acc: 34.786% (17045/49000)
[Evaluate] Valid. Acc.: 41.200%          Test Acc.: 42.190%
Train Time used: 4       Loss: 1.624 | Train Acc: 42.837% (20990/49000)
[Evaluate] Valid. Acc.: 44.300%          Test Acc.: 43.790%
Train Time used: 4       Loss: 1.502 | Train Acc: 47.312% (23183/49000)
[Evaluate] Valid. Acc.: 48.200%          Test Acc.: 47.810%
Train Time used: 4       Loss: 1.351 | Train Acc: 52.988% (25964/49000)
[Evaluate] Valid. Acc.: 49.500%          Test Acc.: 50.780%
Train Time used: 4       Loss: 1.317 | Train Acc: 53.951% (26436/49000)
[Evaluate] Valid. Acc.: 48.200%          Test Acc.: 47.970%
Train Time used: 5       Loss: 1.431 | Train Acc: 49.918% (24460/49000)
[Evaluate] Valid. Acc.: 43.900%          Test Acc.: 44.960%
Train Time used: 5       Loss: 1.390 | Train Acc: 51.424% (25198/49000)
[Evaluate] Valid. Acc.: 49.700%          Test Acc.: 49.460%
Train Time used: 4       Loss: 1.248 | Train Acc: 56.376% (27624/49000)
[Evaluate] Valid. Acc.: 51.000%          Test Acc.: 51.410%
[Result] Valid. Acc.: 51.000%    Test Acc.: 51.410%

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
from utils import evaluate
from utils.img_jitter import ImageJitter
from scipy.special import softmax


if __name__ == "__main__":

    # config:
    is_jiter = True

    merged_data = get_all_train_data("cifar-10-batches-py")
    train_data, valid_data = data_split(merged_data, n_valid=1000)
    test_data = load_batch("cifar-10-batches-py/test_batch")

    scaler = StandardScaler()
    scaler.fit(train_data['pixel_data'])
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
    net = TwoLayerNeuralNetwork(n_hidden_nodes=[50], p_dropout=0.0)
    # net = TwoLayerNeuralNetwork(n_hidden_nodes=[768])
    ntrain = train_data['labels'].shape[0]
    n_step_per_cycle = 2
    ncycle = 2
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
    jitr = ImageJitter()
    for epoch in range(n_epoch):
        loss = 0
        total = 0
        correct = 0
        st = time.time()
        net.train()
        for inputs, labels in train_loader:
            # jitter the images on-fly
            if is_jiter:
                inputs = jitr.jitter_batch(inputs)
            inputs = scaler.transform(inputs)
            out = net(inputs)
            grads = net.backward(out, labels, weight_decay)
            loss += net.cross_entropy(out, labels)
            if weight_decay > 0:
                loss += weight_decay * net.L2_penalty()
            net.update(grads, lrate=scheduler.get_lr())
            scheduler.step()
            # ============================================
            # make prediction
            # apply softmax
            s_mat = softmax(out, axis=0)
            # obtain the top one
            pred = np.argmax(s_mat, axis=0)
            correct += np.sum(pred == labels)
            total += labels.shape[0]

        # print stats
        used_time = time.time() - st
        loss /= total
        acc = correct / total
        print('Train Time used: %d \t Loss: %.3f | Train Acc: %.3f%% (%d/%d)' %
            (used_time, loss, acc*100, correct, total))

        # =====================================================================
        valid_acc = evaluate(valid_loader, net)
        test_acc = evaluate(test_loader, net)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            corrs_test_acc = test_acc

        print("[Evaluate] Valid. Acc.: %.3f%% \t Test Acc.: %.3f%%" % (
            valid_acc*100, test_acc*100))

    print("[Result] Valid. Acc.: %.3f%% \t Test Acc.: %.3f%%" % (
        best_valid_acc*100, corrs_test_acc*100))
