from utils.text_preproc import TextLinesReader
from clsr.rnn import VanillaRNN
from scipy.special import softmax
import numpy as np
import utils

class AdaGradOptim:
    """
    AdaGrad optimization scheme. For RNN only now
    """

    def __init__(self, model, lr=0.1):
        self.model = model
        self.params_names = model.parameters()
        self.lr = lr
        self.eps = np.finfo(model.dtype).eps
        # the accumulative squared gradients
        self._accu_sq_grad = dict()

    def backward(self, logits, targets):
        grads = self.model._get_backward_grad(logits, targets, clipping=True)

        # update the momentum values
        self._update_ada_grads(grads)

        # update the model parameters with grads
        self._backward_with_grads(grads)


    def _update_ada_grads(self, grads):
        for k in self.params_names:
            grad_name = "grad_" + k
            grad = grads[grad_name]
            if k in self._accu_sq_grad:
                self._accu_sq_grad[k] = self._accu_sq_grad[k] + grad**2
            else:
                # first time
                self._accu_sq_grad[k] = grad**2

    def _backward_with_grads(self, grads):
        for k in self.params_names:
            theta = getattr(self.model, k)  # obtain the parameter reference
            grad = grads["grad_" + k]
            sq_grad = self._accu_sq_grad[k]
            update_amt = self.lr * grad / np.sqrt(sq_grad + self.eps)
            theta -= update_amt



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