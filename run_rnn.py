from utils.text_preproc import TextLinesReader
from clsr.rnn import VanillaRNN
from scipy.special import softmax
import numpy as np
import utils

class AdaGradOptim:
    """
    AdaGrad optimization scheme. For RNN only now
    """

    def __init__(self, model, params_names):
        self.params_names = params_names
        self.momentum = dict()
        for k in params_names:
            self.momentum[k] = None

    def update(self):
        pass

    def _backward(self):
        pass


if __name__ == "__main__":
    with open("rnn_data/goblet_book.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()

    reader = TextLinesReader(lines)
    reader.process()
    seq = reader.get_seq()


    for part_seq in utils.chunks(seq, 25):
        print(part_seq)
        print(len(part_seq))

        # print(len(list(part_seq)))
    # for epochs in range(1):
    #     pass