from utils.load_batch import cifar10_DataLoader
from utils.load_batch import load_batch
from clsr.nn_kl import KLayerNeuralNetwork
import numpy as np
from numpy.testing import assert_allclose

def max_relative_err(grad1, grad2):
    eps = np.finfo(grad1.dtype).eps
    dom = np.abs(grad1 - grad2)
    demon = np.maximum(eps, np.abs(grad1) + np.abs(grad2))
    ret = dom / demon
    return np.max(ret)

def check_num_grad(net, inputs, targets, check_params=None):
    if check_params is None:
        check_params = net.parameters()
    # calculate the gradient analytically
    net.train()
    out = net(inputs)
    grads = net._get_backward_grad(out, targets, weight_decay=0.0)
    # calculate the numerical gradient
    h = 1e-5
    h_inv = 1.0 / h
    net.eval()
    for param in check_params:
        thetas = getattr(net, param)
        for i, theta in enumerate(thetas):
            grad_theta_num = np.zeros(theta.shape)
            for idx in np.ndindex(theta.shape):
                old_val = theta[idx]
                theta[idx] = old_val - h
                out = net(inputs)
                l1 = net.cross_entropy(out, targets)

                theta[idx] = old_val + h
                out = net(inputs)
                l2 = net.cross_entropy(out, targets)
                grad = (l2-l1) * 0.5 * h_inv
                grad_theta_num[idx] = grad
            # -----------------------------------------
            # obtain the analytical form
            grad_an = grads['grad_' + param][i]
            assert grad_an.shape == grad_theta_num.shape

            # do some checking
            name = "grad_%s.%d" % (param, i)
            print(name, max_relative_err(grad_theta_num, grad_an))
            # assert_allclose(grad_theta_num, grad_an, atol=1e-3, rtol=1e-4)


if __name__ == "__main__":
    test_data = load_batch("cifar-10-batches-py/test_batch")
    batch_size = 10
    test_loader = cifar10_DataLoader(test_data, batch_size=batch_size)

    test_inputs = None
    test_labels = None
    for inputs, labels in test_loader:
        test_inputs = inputs[:10, :]
        test_labels = labels
        break
    # print(test_inputs.shape)
    # print(test_labels.shape)


    net = KLayerNeuralNetwork(
        p_dropout=0.0,
        n_features=10,
        n_hidden_nodes=[10, 10], batch_norm=False, dtype='float64')
    check_num_grad(net, test_inputs, test_labels)
    net2 = KLayerNeuralNetwork(
        p_dropout=0.0,
        n_features=10,
        n_hidden_nodes=[10], batch_norm=True, dtype='float64')

    check_num_grad(net2, test_inputs, test_labels, check_params=['bn_shifts', 'bn_scales'])
