"""
Example code to visualize the dataset
Ref: https://stackoverflow.com/a/40144107/8654623
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')

    # convert the key from python2 string to python3 string
    ret = dict()
    for k in data.keys():
        ret[k.decode("utf-8")] = data[k]
    return ret


if __name__ == '__main__':
    data = unpickle("cifar-10-batches-py/data_batch_1")
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
