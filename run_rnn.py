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

    rnn = VanillaRNN()
    optim = AdaGradOptim(rnn)
    for part_seq in utils.chunks(seq, 25):
        print(part_seq)
        print(len(part_seq))

        # print(len(list(part_seq)))
    # for epochs in range(1):
    #     pass