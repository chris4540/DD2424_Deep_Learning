"""
Perform the sanity check mentioned in the question

1. Try to over fit the model
2. The starting loss is around 2.24 and it is reasonable
"""
import numpy as np
from utils.load_batch import cifar10_DataLoader
from utils.load_batch import load_batch
from utils.handle_data import data_split
from utils import train
from utils.lrate import CyclicLR
from clsr.nn_kl import KLayerNeuralNetwork

if __name__ == "__main__":
    train_data = load_batch("cifar-10-batches-py/data_batch_1")
    net = KLayerNeuralNetwork(n_hidden_nodes=[20, 20], p_dropout=0.0, batch_norm=True)

    batch_size = 100
    train_loader = cifar10_DataLoader(train_data, batch_size=batch_size)

    ntrain = train_data['labels'].shape[0]
    n_step_per_cycle = 2
    ncycle = 100
    n_epoch = ncycle*n_step_per_cycle*2
    iter_per_epoch = int(np.ceil(ntrain / batch_size))
    step_size = n_step_per_cycle*iter_per_epoch
    weight_decay = 0.005
    scheduler = CyclicLR(eta_min=1e-5, eta_max=1e-2, step_size=step_size)
    for epoch in range(n_epoch):
        train(train_loader, net, 0.0, scheduler)
