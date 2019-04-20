"""
Plot the learning rate defined in assignment 2 exercise 3.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.lrate import cyc_lrate

if __name__ == '__main__':
    iters = np.arange(1500)
    n_s = 500
    eta_min = 1e-5
    eta_max = 1e-1

    lrates = cyc_lrate(iters, eta_min, eta_max, n_s)

    plt.plot(iters, lrates)
    plt.savefig("assign2/cyc_lrates_ex2.png")

