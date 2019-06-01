"""
https://zhuanlan.zhihu.com/p/38200980

https://wiseodd.github.io/techblog/2016/06/25/dropout/
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

def train(loader, net, lrates):
    pass

def evaluate(loader, net):
    net.eval()
    correct = 0
    total = 0
    for inputs, labels in loader:
        pred = net.predict(inputs)
        correct += np.sum(pred == labels)
        total += labels.shape[0]

    ret = correct/total
    return ret

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
    train_loader = cifar10_DataLoader(train_data, batch_size=100)
    valid_loader = cifar10_DataLoader(valid_data, batch_size=100)
    test_loader = cifar10_DataLoader(test_data, batch_size=100)
    # ==================================================================
    net = TwoLayerNeuralNetwork(n_hidden_nodes=[2000], p_dropout=0.0)
    # net = TwoLayerNeuralNetwork(n_hidden_nodes=[768])
    ntrain = train_data['labels'].shape[0]
    k = 2
    ncycle = 3
    n_epoch = ncycle*k
    # n_epoch = 1
    step_size = k*np.floor(ntrain/100)
    # weight_decay = 0.005
    weight_decay = 0.000
    cyc_lr = CyclicLR(eta_min=1e-5, eta_max=1e-1, step_size=step_size)

    for epoch in range(n_epoch):
        loss = 0
        cnt = 0
        st = time.time()
        for inputs, labels in train_loader:
            net.train()
            out = net(inputs)
            grads = net.backward(out, labels, weight_decay)
            loss += net.cross_entropy(out, labels)
            if weight_decay > 0:
                loss += weight_decay * net.L2_penalty()
            cnt += labels.shape[0]
            net.update(grads, lrate=cyc_lr.get_lr())
            cyc_lr.step()

        loss /= cnt
        endtime = time.time() - st

        vscore = evaluate(valid_loader, net)
        score = evaluate(test_loader, net)
        print(endtime, loss, vscore, score)
