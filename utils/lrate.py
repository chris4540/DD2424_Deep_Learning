import numpy as np
from abc import ABC, abstractmethod

def cyc_lrate(n_iters, eta_min=0, eta_max=1, step_size=5):
    iters = np.arange(n_iters)
    cycle = np.floor(1 + iters/(2*step_size))
    x = np.abs(iters/step_size - 2*cycle + 1)
    eta_t = eta_min + (eta_max-eta_min)*np.maximum(0, (1-x))
    return eta_t

class BaseOptimizer(ABC):


    @abstractmethod
    def step(self):
        raise NotImplementedError("This method should be implemented by the inherited class")

    @abstractmethod
    def get_lr(self):
        raise NotImplementedError("This method should be implemented by the inherited class")


class StepLR(BaseOptimizer):
    """
    Mimic torch.optim.lr_scheduler.StepLR

    Usage:
    >>> scheduler = StepLR(0.05, 30, gamma=1/10)
    >>> for input, labels in loader:
            lrate = scheduler.get_lr()
        scheduler.step()
    """

    def __init__(self, init_lr, step_size, gamma=0.1):
        self.init_lr = init_lr
        self.step_size = step_size
        self.gamma = gamma
        self.total_step = 0

    def step(self):
        self.total_step += 1

    def get_lr(self):
        n_step = self.total_step // self.step_size
        ret = self.init_lr * ((self.gamma)**(-n_step))
        return ret

class CyclicLR(BaseOptimizer):
    """
    Mimic torch.optim.lr_scheduler.CyclicLR

    Usage:
    >>> cyc_lr = CyclicLR(eta_min=1e-5, eta_max=1e-2, step_size=100)
    >>> for input, labels in loader:
            lrate = cyc_lr.get_lr()
            cyc_lr.step()
    """
    def __init__(self, eta_min, eta_max, step_size=2000):
        # force to int
        step_size = int(step_size)
        # obtain one cyclic learning rate
        self.lrates = cyc_lrate(step_size*2, eta_min, eta_max, step_size)
        self.idx = 0

    def step(self):
        self.idx += 1
        if self.idx >= len(self.lrates):
            # reset index
            self.idx = 0

    def get_lr(self):
        return self.lrates[self.idx]

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
