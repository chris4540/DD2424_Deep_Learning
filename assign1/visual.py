"""
Example code to visualize the dataset
Ref: https://stackoverflow.com/a/40144107/8654623
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from load_batch import unpickle

def plot_sample_img(file):
    data = unpickle(file)
    A = data['data']
    Y = data['labels']

    # unflatten the image
    X = A.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    Y = np.array(Y)

    # plot
    fig, axes1 = plt.subplots(5,5,figsize=(3,3))
    for j in range(5):
        for k in range(5):
            i = np.random.choice(range(len(X)))
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(X[i:i+1][0])
    plt.show()

def plot_weight_mat(network):
    W_mat = network.W_mat
    nclass = W_mat.shape[0]

    W_mat = W_mat.reshape(nclass, 3, 32, 32).transpose(0,2,3,1)

    fig, axs = plt.subplots(2, 5, figsize=(5,2), dpi=80)
    for i in range(nclass):
        im = W_mat[i, :]
        # rescale the image
        im_rescale =  (im - np.min(im)) / (np.max(im) - np.min(im))

        j = i // 5
        k = i % 5
        axs[j][k].axis("off")
        axs[j][k].imshow(im_rescale)
    return plt


def plot_loss(network):
    x = list(range(1, len(network.train_costs)+1))
    plt.plot(x, network.train_costs, label='training loss')
    plt.plot(x, network.valid_costs, label='validation loss')
    plt.ylim(1.6, 2.1)
    plt.xlim(left=1)
    plt.legend(loc='upper right')
    plt.title("Plot training and validation loss at each epoch")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    return plt

if __name__ == '__main__':
    plot_sample_img("cifar-10-batches-py/data_batch_1")
