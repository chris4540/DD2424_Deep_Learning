"""
Check numerical grad
"""
from utils.text_preproc import TextLinesReader
from clsr.rnn import VanillaRNN
from scipy.special import softmax
import numpy as np


def max_relative_err(grad1, grad2):
    eps = np.finfo(grad1.dtype).eps
    dom = np.abs(grad1 - grad2)
    demon = np.maximum(eps, np.abs(grad1) + np.abs(grad2))
    ret = dom / demon
    return np.max(ret)

def check_num_grad(rnn, inputs, targets):
    h = 1e-5
    # =====================================================
    h_inv = 1.0 / h
    # compute analyical grad
    out, _ = rnn(inputs)
    grads = rnn._get_backward_grad(out, targets, clipping=False)

    rnn.eval()
    for k in grads.keys():
        attrname = k.split("grad_")[1]
        theta = getattr(rnn, attrname)  # obtain the arr in the rnn
        grad_theta_num = np.zeros(theta.shape)
        for idx in np.ndindex(theta.shape):
            old_val = theta[idx]
            theta[idx] = old_val - h
            out, _ = rnn(inputs)
            l1 = rnn.cross_entropy(out,targets)

            theta[idx] = old_val + h
            out, _ = rnn(inputs)
            l2 = rnn.cross_entropy(out, targets)
            grad = (l2-l1) * 0.5 * h_inv
            grad_theta_num[idx] = grad
    # ===========================================================
        print(k, max_relative_err(grad_theta_num, grads[k]))
        print(k, np.allclose(grad_theta_num, grads[k], atol=1e-3, rtol=1e-4))

if __name__ == "__main__":
    with open("rnn_data/goblet_book.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()

    reader = TextLinesReader(lines)
    reader.process()
    seq = reader.get_seq()

    rnn = VanillaRNN(n_hidden_node=5, dtype='float64')
    inputs = seq[:25]
    targets = seq[1:26]
    check_num_grad(rnn, inputs, targets)