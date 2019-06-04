from utils.text_preproc import TextLinesReader
from clsr.rnn import VanillaRNN
from scipy.special import softmax
import numpy as np
from numpy.testing import assert_allclose


# def getRelativeErrors(grad1, grad2):
#     """
#     Computes the relative errors of grad_1 and grad_2 gradients
#     """
#     abs_diff = np.absolute(grad1 - grad2)
#     abs_sum = np.absolute(grad1) + np.absolute(grad2)
#     max_elems = np.where(abs_sum > np.finfo(float).eps, abs_sum, np.finfo(float).eps)
#     relativeErrors = abs_diff / max_elems
#     return relativeErrors

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

    rnn = VanillaRNN(n_hidden_node=5)
    # syn_seq = rnn.synthesize_seq(seq_oh[:, 0], length=10)
    # chars = reader.map_idxs_to_chars(syn_seq)
    # print(''.join(chars))
    # =============================================================
    # inputs
    inputs = seq[:25]
    outputs = seq[1:26]

    input_oh = reader.get_one_hot(inputs)
    target_oh = reader.get_one_hot(outputs)
    # rnn.eval()
    out = rnn(input_oh)
    # print(out)
    loss = rnn.cross_entropy(out, np.array(outputs))
    # print(loss)

    grads = rnn._get_backward_grad(out, target_oh, clipping=False)
    print(grads['grad_U'].dtype)

    # =========================================================================
    rnn.eval()
    h = 1e-4
    h_recip = 1.0 / h
    grad_U = np.zeros(rnn.input_wgts.shape, dtype='float32')
    init_hidden = np.zeros((5, 1), dtype='float32')

    for idx in np.ndindex(rnn.input_wgts.shape):
        old_val = rnn.input_wgts[idx]

        rnn.input_wgts[idx] = old_val + h
        out = rnn(input_oh, init_hidden)
        l1 = rnn.cross_entropy(out, np.array(outputs))
        # ==================================================
        rnn.input_wgts[idx] = old_val - h
        out = rnn(input_oh, init_hidden)
        l2 = rnn.cross_entropy(out, np.array(outputs))
        # print(l1, l2)
        grad_U[idx] = (l2-l1) / (2*h)

        rnn.input_wgts[idx] = old_val
    eps = np.finfo('float32').eps
    dom = np.abs(grad_U - grads['grad_U'])
    demon = np.maximum(eps, np.abs(grad_U) + np.abs(grads['grad_U']))
    print(np.max(dom/ demon))
    # print(np.allclose(grads['grad_U'], grad_U))
    # print(grads['grad_U'])
    # print(grad_U)
    # getRelativeErrors(grad_U, grads['grad_U'])
    # # p_mat = softmax(out, axis=0)
    # # rnn._get_backward_grad(out, target_oh)

