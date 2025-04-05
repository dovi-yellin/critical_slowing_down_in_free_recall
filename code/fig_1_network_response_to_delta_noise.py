import copy
import sys

import pickle
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab

from initParams import initParams, read_config

# Dovi - leave the following as examples for future analysis code
from models import RateModel
#from RateModelAnalysis import neural_activity_plot, stable_activity_plot
#from RateModelAnalysis import welch_spectrum_plot, multi_run_analysis, compute_sum_power

if __name__ == '__main__':
    config = read_config('test_general.json')
    random_seed = int(config['random_seed'])
    np.random.seed(random_seed)
    t_stabilize_time = 1 # int(config['params']['stabilize_time'])  # 400
    t_total_time = 10 # int(config['params']['total_time'])          # 1400
    t_N = 10 # int(config['params']['N'])                            # 400
    tau = float(config['params']['tau'])                        # 20.0
    alpha = float(config['params']['alpha'])                    # 0.05
    wins = int(config['params']['wins'])                        # 24000

    # initialize simulation 1 params
    params = initParams(N=t_N, stabilize_time=t_stabilize_time, total_time=t_total_time)
    params.tau = tau

    gamma_subcritical = 0.09
    gamma_critical = 0.108
    gamma_supercritical = 0.112

    params.gamma = gamma_supercritical # float(config['params']['gamma1'])

    params.t_input_off = 1 # define length of short delta noise

    gamma = params.gamma
    mu = params.mu
    probability = params.prb

    results_dicts = []

    control_param = gamma * mu * probability

    model = RateModel(params=params)

    sys.stdout.flush()  # ensures progress bar starts after print
    print(f"Beginning network simulation with {gamma=:.6f}, {mu=}, {probability=}, {control_param=:.6f}")
    #print(f"{N=}")
    sys.stdout.flush()  # ensures progress bar starts after print

    # run the model simulation
    results_dict = model.run_local_circuit()

    p = results_dict['params']
    N = p.N
    fs = p.fs
    r_store = results_dict['r_store']
    start_sample = int(p.start_sample)
    activity_per_unit = np.transpose(r_store)

    sig_len = len(activity_per_unit[0, :])
    xvec = np.linspace(0, int(sig_len / fs), sig_len)

    lettersize = 24
    # plot activity per unit over first n units
    fig = plt.figure(1, figsize=(15, 5))
    N = 100 if N > 100 else N
    for i in range(0, N):
        plt.plot(xvec, activity_per_unit[i, :])
        plt.xlabel('Time (sec)', fontsize=lettersize)
        plt.ylabel('Activity rate', fontsize=lettersize)
    plt.show()
    plt.close()

    mean_activity = np.mean(activity_per_unit, axis=0)
    mean_activity_norm = mean_activity / np.max(mean_activity)

    # plot mean activity
    plt.plot(xvec, mean_activity_norm)
    plt.xlabel('Time (sec)', fontsize=lettersize)
    plt.ylabel('Activity rate (normalized)', fontsize=lettersize)

    plt.yticks(np.arange(0, 2, 1))
    plt.ylim(0.00001, 1.0)
    plt.xticks(fontsize=lettersize)
    plt.yticks(fontsize=lettersize)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.show()
    fig.tight_layout()
    plt.close()

    mean_activity = mean_activity - np.mean(mean_activity)

    # plot the Welch power spectrum
    s, fr = mlab.psd(mean_activity, NFFT=int(mean_activity.size / 2), Fs=fs)
    if fig is None:
        fig = plt.figure(figsize=(6, 5))
    plt.loglog(fr, s)
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('Freq. (Hz)', fontsize=lettersize)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylabel('Power (index)', fontsize=lettersize)
    fig.tight_layout()

    Fs = params.fs

    spct_per_sample = np.abs(np.fft.rfft(activity_per_unit, axis=1))
    spct = np.mean(spct_per_sample, axis=0)

    freqs = np.fft.fftfreq(spct.size, d=1. / Fs)
    freqs = np.arange(0, Fs, Fs / spct.size)

    plt.loglog(freqs, spct)
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('Freq. (Hz)', fontsize=lettersize)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylabel('Power (index)', fontsize=lettersize)
    fig.tight_layout()
    plt.show()


    b_save = True
    results_dir='results'
    if b_save:
        iso_8601_format = '%Y%m%dT%H%M%S'  # e.g., 20211119T221000
        fname = f"{results_dir}/run_{datetime.now().strftime(iso_8601_format)}.pkl"
        print(f"dumping results to {fname}")
        with open(fname, 'wb') as f:
            pickle.dump(results_dict, f) # pickle.dump(mean_pwr_arr, f)

    Pause = 0.1