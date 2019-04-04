"""
This file is for the Exercise 1 described in the assignment1.pdf
"""
from one_layer_ann import OneLayerNetwork
from load_batch import load_batch

if __name__ == '__main__':
    # get the training data
    train_data = load_batch("cifar-10-batches-py/data_batch_1")

    # network = OneLayerNetwork(10, 20, lambda_=0.0)
    X_mat = train_data["pixel_data"]
    y_mat = train_data["onehot_labels"]

    # # p = network.evaluate(X_mat)

    # # cost = network.compute_cost(X_mat, y_mat)
    # grad_W, grad_b = network.compute_grad(X_mat[:20, :5], y_mat[:, :5])
    # grad_W2, grad_b2 = network.compute_grads_num(X_mat[:20, :5], y_mat[:, :5])

    # print("Is grad_W all close:", np.allclose(grad_W, grad_W2))
    # print("Is grad_b all close:", np.allclose(grad_b, grad_b2))
