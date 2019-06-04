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
    rnn = VanillaRNN(n_hidden_node=n_hidden)
    optim = AdaGradOptim(rnn)
    smooth_loss = None
    n_iter = 0
    for epoch in range(2):
        # define the generator
        input_gen = utils.window(seq, 25)
        targets_gen = utils.window(seq[1:], 25)
        h_prev = np.zeros((n_hidden, 1), dtype='float32')
        for inputs, targets in zip(input_gen, targets_gen):
            # forward
            logits = rnn(inputs, h_prev)

            # backward
            optim.backward(logits, targets)

            # calculate the loss
            loss = rnn.cross_entropy(logits, targets)
            if smooth_loss is None:
                smooth_loss = loss
            else:
                smooth_loss = .999*smooth_loss + .001*loss

            n_iter += 1

            # print loss every 100
            if n_iter % 1000 == 0:
                print("iteration: %d \t  smooth_loss: %f" % (n_iter, smooth_loss))
            if n_iter % 10000 == 0:
                seq = rnn.synthesize_seq(inputs[0], h_0=h_prev, length=200)
                # translate it
                char_seq = reader.map_idxs_to_chars(seq)
                print(''.join(char_seq))
