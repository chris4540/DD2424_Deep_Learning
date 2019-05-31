import numpy as np

def cyc_lrate(n_iters, eta_min=0, eta_max=1, step_size=5):
    iters = np.arange(n_iters)
    cycle = np.floor(1 + iters/(2*step_size))
    x = np.abs(iters/step_size - 2*cycle + 1)
    eta_t = eta_min + (eta_max-eta_min)*np.maximum(0, (1-x))
    return eta_t

class CyclicLR:
    """
    Mimic torch.optim.lr_scheduler.CyclicLR

    Usage:
    >>> cyc_lr = CyclicLR(eta_min=1e-5, eta_max=1e-2, step_size=100)
    >>> for input, labels in loader:
            lrate = cyc_lr.get_lr()
            cyc_lr.step()
    """
    def __init__(self, eta_min, eta_max, step_size=2000):
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