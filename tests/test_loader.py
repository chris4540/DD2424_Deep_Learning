import numpy as np
import utils
import utils.load_batch
from utils.load_batch import load_batch
from utils.load_batch import cifar10_DataLoader
from utils.handle_data import data_split
from utils.handle_data import get_all_train_data
from utils.preprocess import StandardScaler
from clsr.nn_2l import TwoLayerNeuralNetwork


if __name__ == "__main__":
    merged_data = get_all_train_data("cifar-10-batches-py")
    test_data = load_batch("cifar-10-batches-py/test_batch")
    loader = cifar10_DataLoader(test_data, batch_size=100)
    cnt = 0
    for inputs, labels in loader:
        cnt += labels.shape[0]
    print(cnt)

    cnt = 0
    for inputs, labels in loader:
        cnt += labels.shape[0]
    print(cnt)