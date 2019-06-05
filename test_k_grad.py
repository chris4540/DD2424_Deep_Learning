from utils.load_batch import cifar10_DataLoader
from utils.load_batch import load_batch
from clsr.nn_kl import KLayerNeuralNetwork
import numpy as np

def max_relative_err(grad1, grad2):
    eps = np.finfo(grad1.dtype).eps
    dom = np.abs(grad1 - grad2)
    demon = np.maximum(eps, np.abs(grad1) + np.abs(grad2))
    ret = dom / demon
    return np.max(ret)


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
        n_hidden_nodes=[10], batch_norm=False, dtype='float64')
    net.train()
    out = net(test_inputs)
    grads = net._get_backward_grad(out, test_labels, weight_decay=0.0)
    # ========================================================================
    # print(grads)
    # net.eval()
    for param in net.parameters():
        thetas = getattr(net, param)
        for i, theta in enumerate(thetas):
            grad_theta_num = np.zeros(theta.shape)
            h = 1e-5
            h_inv = 1.0 / h
            for idx in np.ndindex(theta.shape):
                old_val = theta[idx]
                theta[idx] = old_val - h
                out = net(test_inputs)
                l1 = net.cross_entropy(out, test_labels)

                theta[idx] = old_val + h
                out = net(test_inputs)
                l2 = net.cross_entropy(out, test_labels)
                grad = (l2-l1) * 0.5 * h_inv
                grad_theta_num[idx] = grad
            # -----------------------------------------
            # obtain the analytical form
            grad_an = grads['grad_' + param][i]
            assert grad_an.shape == grad_theta_num.shape
            name = "grad_%s.%d" % (param, i)
            print(name, max_relative_err(grad_theta_num, grad_an))

