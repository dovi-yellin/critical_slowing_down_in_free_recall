""" Try to analytically fit power spectrum from Norman & Malach paper using a Lorenzian function's params.
@author: Dovi Yellin, Noam Siegel
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
from matlab_to_numpy import loadmat

K = 5.05
tau = 20  # 0.3
alpha = 0.0025
f_fast = 8.0
lamda_fast = 2 * np.pi * f_fast
f_slow = 0.41 # 0.08
lamda_slow = 2 * np.pi * f_slow
if __name__ == '__main__':
    freq = np.arange(0.01, 50, 0.01)
    # spct1 = 0.54 * (0.01 / freq + 1.0 / (freq + 5.0))
    spct1 = K * ((alpha / (freq*freq + f_slow*f_slow)) + (1.0 - alpha) / (freq*freq + f_fast*f_fast))
    spct_db1 = 10 * np.log10(spct1)
    #spct2 = 0.47 * (0.01 / freq + 1.0 / (freq + 4.0))
    f_slow -= 0.08
    spct2 = K * ((alpha / (freq * freq + f_slow * f_slow)) + (1.0 - alpha) / (freq * freq + f_fast * f_fast))
    spct_db2 = 10 * np.log10(spct2)
    # fetch Norman et al data
    fpath = r"C:\Research\Spontaneous_activity\Rate_model\CriticalSlowDown\data\figdata.mat"
    d = loadmat(fpath)
    X = np.asarray(d['figdata']['rawSpectrum']['X'])
    Y = np.asarray(d['figdata']['rawSpectrum']['Y'])
    fig = plt.figure(1, figsize=(5, 5))
    lettersize = 24
    plt.semilogx(freq, spct_db1, label='Lorenzian - Sub Critical', linewidth=3.5, color='green')  # plt.plot(freqs, spct_db)
    plt.semilogx(freq, spct_db2, label='Lorenzian - Near Critical', linewidth=3.5, color='orange')
    plt.semilogx(X[0], Y[0], label='Norman - Recall', linewidth=3, alpha=0.5, color='red')  # plt.plot(X[0], Y[0])
    plt.semilogx(X[1], Y[1], label='Norman - Resting state', linewidth=3, alpha=0.5, color='blue')
    plt.xlabel('Freq. (Hz)', fontsize=lettersize)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylabel('Power (dB)', fontsize=lettersize)
    plt.legend(loc='best')
    plt.xlim(0.01, 15)
    plt.ylim(-20, -5)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    fig.tight_layout()

    plt.show()