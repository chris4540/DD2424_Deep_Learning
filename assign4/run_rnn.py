from utils.text_preproc import TextLinesReader
from clsr.rnn import VanillaRNN
from scipy.special import softmax
import numpy as np
from utils.lrate import AdaGradOptim

if __name__ == "__main__":
    with open("rnn_data/goblet_book.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()

    reader = TextLinesReader(lines)
    reader.process()
    seq = reader.get_seq()

    n_hidden = 100
    seq_length = 25
    rnn = VanillaRNN(n_hidden_node=n_hidden, dtype='float32')
    optim = AdaGradOptim(rnn)
    smooth_loss = None
    n_iter = 0
    best_loss = np.inf
    h_prev = None
    for epoch in range(10):
        h_prev = np.zeros((n_hidden, 1), dtype='float32')
        for e in range(0, len(seq), seq_length):
            inputs = seq[e:e+seq_length]
            targets = seq[e+1:e+seq_length+1]
            if e + seq_length > len(seq):
                inputs = seq[e:len(seq)-2]
                targets = seq[e+1:len(seq)-1]

            logits, h_next = rnn(inputs, h_prev)

            # backward
            optim.backward(logits, targets)

            # calculate the loss
            loss = rnn.cross_entropy(logits, targets)
            if smooth_loss is None:
                smooth_loss = loss
            else:
                smooth_loss = .999*smooth_loss + .001*loss

            # check if save
            if smooth_loss < best_loss:
                best_loss = smooth_loss
                best_state = rnn.state_dict()

            n_iter += 1

            # print loss every 100
            if n_iter % 1000 == 0:
                print("iteration: %d \t  smooth_loss: %f" % (n_iter, smooth_loss))

            if n_iter % 10000 == 0:
                input_char = [inputs[0]]
                syn_seq = rnn.synthesize_seq(input_char, h_0=h_prev, length=200)
                # translate it
                print("Generated text:")
                print("----------------------")
                print(''.join(reader.map_idxs_to_chars(syn_seq)))


            h_prev = h_next
    # =======================================================================
    # write final essay
    rnn.load_state_dict(best_state)
    inputs = reader.map_chars_to_idxs(['.'])
    syn_seq = rnn.synthesize_seq(inputs, h_0=h_prev, length=1000)
    print("final essay:")
    print("----------------------")
    print(''.join(reader.map_idxs_to_chars(syn_seq)))
