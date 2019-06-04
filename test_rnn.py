from utils.text_preproc import TextLinesReader
from clsr.rnn import VanillaRNN
from scipy.special import softmax
import numpy as np
from numpy.testing import assert_allclose

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
    out = rnn(inputs)
    grads = rnn._get_backward_grad(out, targets, clipping=False)

    rnn.eval()
    for k in grads.keys():
        attrname = k.split("grad_")[1]
        theta = getattr(rnn, attrname)  # obtain the arr in the rnn
        grad_theta_num = np.zeros(theta.shape)
        for idx in np.ndindex(theta.shape):
            old_val = theta[idx]
            theta[idx] = old_val - h
            out = rnn(inputs)
            l1 = rnn.cross_entropy(out,targets)

            theta[idx] = old_val + h
            out = rnn(inputs)
            l2 = rnn.cross_entropy(out, targets)
            grad = (l2-l1) * 0.5 * h_inv
            grad_theta_num[idx] = grad
    # ===========================================================
        print(k, max_relative_err(grad_theta_num, grads[k]))

if __name__ == "__main__":
    with open("rnn_data/goblet_book.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()

    reader = TextLinesReader(lines)
    reader.process()
    seq = reader.get_seq()

    # print(len(seq))

    # test_seq = seq[:10]
    # print(test_seq)

    # seq_oh = reader.get_one_hot(test_seq)
    # print(seq_oh.shape)

    rnn = VanillaRNN(n_hidden_node=5, dtype='float64')
    # syn_seq = rnn.synthesize_seq(seq_oh[:, 0], length=10)
    # chars = reader.map_idxs_to_chars(syn_seq)
    # print(''.join(chars))
    # =============================================================
    # inputs
    inputs = seq[:25]
    targets = seq[1:26]

    # # rnn.eval()
    # out = rnn(inputs)
    # prob = softmax(out, axis=0)
    # # print(out)
    # loss = rnn.cross_entropy(out, np.array(targets))
    # # print(loss)

    # grads = rnn._get_backward_grad(out, targets, clipping=False)
    # print(grads['grad_c'].dtype)

    check_num_grad(rnn, inputs, targets)

    # =========================================================================
    # rnn.eval()
    # h = 1e-3
    # h_recip = 1.0 / h
    # grad_c = np.zeros(rnn.output_bias.shape, dtype='float32')
    # init_hidden = np.zeros((5, 1), dtype='float32')

    # for idx in np.ndindex(rnn.output_bias.shape):
    #     old_val = rnn.output_bias[idx]
    #     rnn.output_bias[idx] = old_val - h
    #     out = rnn(input_oh, init_hidden)
    #     l1 = rnn.cross_entropy(out, np.array(targets))
    #     # ==================================================
    #     rnn.output_bias[idx] = old_val + h
    #     out = rnn(input_oh, init_hidden)
    #     l2 = rnn.cross_entropy(out, np.array(targets))
    #     print(l1, l2)
    #     # print(l1, l2)
    #     grad_c[idx] = (l2-l1) / (2*h)

    #     rnn.output_bias[idx] = old_val

