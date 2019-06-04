from utils.text_preproc import TextLinesReader
from clsr.rnn import VanillaRNN
from scipy.special import softmax
import numpy as np
# from utils import one_hot




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

    rnn = VanillaRNN()
    # syn_seq = rnn.synthesize_seq(seq_oh[:, 0], length=10)
    # chars = reader.map_idxs_to_chars(syn_seq)
    # print(''.join(chars))
    # =============================================================
    # inputs
    inputs = seq[:10]
    outputs = seq[1:11]

    input_oh = reader.get_one_hot(inputs)
    target_oh = reader.get_one_hot(outputs)
    out = rnn(input_oh)
    # p_mat = softmax(out, axis=0)
    rnn._get_backward_grad(out, target_oh)

