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
from utils.lrate import CyclicLR
from clsr.nn_kl import KLayerNeuralNetwork
from utils.preprocess import StandardScaler

if __name__ == "__main__":
    train_data = load_batch("cifar-10-batches-py/data_batch_1")
    net = KLayerNeuralNetwork(n_hidden_nodes=[50, 50], p_dropout=0.0, batch_norm=False, batch_norm_momentum=.9)

    scaler = StandardScaler()
    scaler.fit(train_data['pixel_data'])
    train_data['pixel_data'] = scaler.transform(train_data['pixel_data'])

    batch_size = 100
    train_loader = cifar10_DataLoader(train_data, batch_size=batch_size, shuffle=False)

    ntrain = train_data['labels'].shape[0]
    n_epoch = 200
    scheduler = StepLR(0.05, 20, gamma=.5)
    # scheduler = CyclicLR(eta_min=1e-5, eta_max=1e-1, step_size=1000)
    for epoch in range(n_epoch):
        train(train_loader, net, 0.0, scheduler)
