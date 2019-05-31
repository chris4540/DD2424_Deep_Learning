"""
https://zhuanlan.zhihu.com/p/38200980
"""
import numpy as np
from utils.load_batch import load_batch
from utils.load_batch import cifar10_dataloader
from utils.handle_data import data_split
from utils.handle_data import get_all_train_data
from utils.preprocess import StandardScaler
from clsr.nn_2l import TwoLayerNeuralNetwork

if __name__ == "__main__":
    merged_data = get_all_train_data("cifar-10-batches-py")
    train_data, valid_data = data_split(merged_data, n_valid=1000)
    test_data = load_batch("cifar-10-batches-py/test_batch")

    # scaler = StandardScaler()
    # scaler.fit(train_data['pixel_data'])
    # train_data['pixel_data'] = scaler.transform(train_data['pixel_data'])
    # valid_data['pixel_data'] = scaler.transform(valid_data['pixel_data'])
    # test_data['pixel_data'] = scaler.transform(test_data['pixel_data'])
    # print("Done preprocessing!")
    # ==================================================================
    net = TwoLayerNeuralNetwork(n_hidden_nodes=[50])
    cnt = 0
    for inputs, labels in cifar10_dataloader(train_data, batch_size=100):
        net.eval()
        out = net(inputs)

        loss1 = net.cross_entropy(out, labels)
        print(loss1/100)
        cnt += labels.shape[0]
        break

    # print(cnt)
    # print(type(train_data['labels']))
    # print(train_data['labels'].shape)
    # # print(test_data.keys())
    # # print(train_data.keys())
    # # print(valid_data.keys())
    # # =================================================================
    # # make the network
    # # clsr = TwoLayerNeuralNetwork()
    # # initialize the weights and bias
    # # clsr.initalize_wgts()
