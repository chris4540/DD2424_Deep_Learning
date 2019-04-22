import numpy as np

def cyc_lrate(n_iters, eta_min=0, eta_max=1, step_size=5):
    iters = np.arange(n_iters)
    cycle = np.floor(1+iters/(2*step_size))
    x = np.abs(iters/step_size - 2*cycle + 1)
    eta_t = eta_min + (eta_max-eta_min)*np.maximum(0, (1-x))
    return eta_t
