from one_layer_ann import OneLayerNetwork
from load_batch import load_batch

if __name__ == '__main__':
    train_data = load_batch("cifar-10-batches-py/data_batch_1")
    valid_data = load_batch("cifar-10-batches-py/data_batch_2")
    test_data = load_batch("cifar-10-batches-py/test_batch")

    nclass = train_data["onehot_labels"].shape[0]
    ndim = train_data["pixel_data"].shape[0]

    ann = OneLayerNetwork(verbose=True)
    ann.set_train_data(train_data["pixel_data"], train_data["onehot_labels"])
    ann.set_valid_data(valid_data["pixel_data"], valid_data["onehot_labels"])
    ann.train()
    acc = ann.compute_accuracy(test_data["pixel_data"], test_data["labels"])
