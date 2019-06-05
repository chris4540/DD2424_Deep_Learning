from utils.text_preproc import TextLinesReader
from clsr.rnn import VanillaRNN
from scipy.special import softmax
import numpy as np
import utils
from utils.lrate import AdaGradOptim

if __name__ == "__main__":
    with open("trump_tweets.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()

    reader = TextLinesReader(lines)
    reader.process()
    seq = reader.get_seq()

    n_hidden = 100
    seq_length = 25

    end_of_tweet = reader.map_chars_to_idxs('#')[0]
    print(len(seq))

    rnn = VanillaRNN(
        n_hidden_node=n_hidden,
        n_features=reader.n_char,
        n_classes=reader.n_char,
        dtype='float32')

    optim = AdaGradOptim(rnn)
    smooth_loss = None
    n_iter = 0
    best_loss = np.inf
    h_prev = None
    for epoch in range(5):
        h_prev = np.zeros((n_hidden, 1), dtype='float32')
        e = 0
        while e < len(seq):
            inputs = seq[e:e+seq_length]
            targets = seq[e+1:e+seq_length+1]
            # take care of end of tweet
            if end_of_tweet in inputs:
                ending = inputs.index(end_of_tweet)
                if ending != 0:
                    # print(ending)
                    inputs = seq[e:e+ending]
                    targets = seq[e+1:e+ending+1]
                    e += ending+1
                    is_end_tweet = True
                else:
                    e += 1
                    continue
            else:
                e += seq_length
                is_end_tweet = False
            # ================================================================
            if len(inputs) == 0 or len(targets) == 0:
                # went to the end
                e += seq_length
                continue

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
                pred = np.argmax(logits, axis=0)
                print("Predict: ", ''.join(reader.map_idxs_to_chars(pred)))
                print("Target:  ", ''.join(reader.map_idxs_to_chars(inputs)))


            if n_iter % 10000 == 0:
                input_char = [inputs[0]]
                # h0 = np.zeros((n_hidden, 1), dtype='float32')
                syn_seq = rnn.synthesize_seq(input_char, h_0=h_prev, length=140)
                # translate it
                print("Generated text:")
                print("----------------------")
                print(''.join(reader.map_idxs_to_chars(syn_seq)))

            if is_end_tweet:
                h_prev = np.zeros((n_hidden, 1), dtype='float32')
            else:
                h_prev = h_next
