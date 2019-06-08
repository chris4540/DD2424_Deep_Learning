"""
For assignment 3 quesiton 1

Perform the sanity check mentioned in the question

1. Try to over fit the model
2. The starting loss is around 2.24 and it is reasonable
3. Use the simplest scheduler i.e. StepLR
4. with / without batchnorm should behave the same
"""
import numpy as np
from utils.load_batch import cifar10_DataLoader
from utils.load_batch import load_batch
from utils.handle_data import data_split
from utils import train
from utils.lrate import StepLR
from clsr.nn_kl import KLayerNeuralNetwork
from utils.preprocess import StandardScaler

if __name__ == "__main__":
    train_data = load_batch("cifar-10-batches-py/data_batch_1")
    # net = KLayerNeuralNetwork(n_hidden_nodes=[100, 50], p_dropout=0.0, batch_norm=False)
    net = KLayerNeuralNetwork(n_hidden_nodes=[80, 50], p_dropout=0.0, batch_norm=True, batch_norm_momentum=0.9)

    scaler = StandardScaler()
    scaler.fit(train_data['pixel_data'])
    train_data['pixel_data'] = scaler.transform(train_data['pixel_data'])

    batch_size = 100
    train_loader = cifar10_DataLoader(train_data, batch_size=batch_size)

    ntrain = train_data['labels'].shape[0]
    n_epoch = 500
    scheduler = StepLR(0.01, 100, gamma=.1)
    for epoch in range(n_epoch):
        train(train_loader, net, 0.0, scheduler)
