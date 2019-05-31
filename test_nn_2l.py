"""
https://zhuanlan.zhihu.com/p/38200980
"""
import numpy as np
from utils.load_batch import load_batch
from utils.load_batch import cifar10_DataLoader
from utils.handle_data import data_split
from utils.handle_data import get_all_train_data
from utils.preprocess import StandardScaler
from clsr.nn_2l import TwoLayerNeuralNetwork


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
    test_loader = cifar10_DataLoader(test_data, batch_size=100)
    # ==================================================================
    net = TwoLayerNeuralNetwork(n_hidden_nodes=[50])
    cnt = 0

    score = evaluate(test_loader, net)
    print(score)
    for inputs, labels in train_loader:
        net.train()
        out = net(inputs)
        grads = net.backward(out, labels)
        net.update(grads, lrate=1e-4)
        cnt += labels.shape[0]

    score = evaluate(test_loader, net)
    print(score)
