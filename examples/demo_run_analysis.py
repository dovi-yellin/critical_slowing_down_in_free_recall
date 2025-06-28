import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import mlab

from initParams import initParams, read_config

path_str = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(path_str, r'results//rate_model_G_0.8_to_0.98_run_20250607T130640.pkl')
results_dicts= pickle.load(open(filename, 'rb'))

params: initParams = results_dicts[0]['params']
N = params.N
fs = params.fs
start_sample = int(params.start_sample)

lettersize = 20
fig = plt.figure(1, figsize=(6, 5))

for idx, results_dict in enumerate(results_dicts):
    params: initParams = results_dict['params']
    control_param = params.gamma * params.mu * params.prb

    r_store = results_dict['r_store']
    activity_per_unit = np.transpose(r_store)
    start_trimmed_activity = activity_per_unit[:, (start_sample - 1):-1]

    mean_activity = np.mean(start_trimmed_activity, axis=0)
    mean_activity = mean_activity - np.mean(mean_activity)

    b_plot_activity = False
    if b_plot_activity:
        fig = plt.figure(2, figsize=(8, 4))
        plt.plot(mean_activity)
        plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x / 1000:g}'))
        plt.xlabel('Time (s)', fontsize=lettersize)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.ylabel('Activity (mean)', fontsize=lettersize)
        plt.tight_layout
        plt.close()

    # plot the Welch power spectrum
    s, fr = mlab.psd(mean_activity, NFFT=int(mean_activity.size / 10), Fs=fs)
    plt.loglog(fr, s, label="%.2f" % control_param)

plt.grid()
#plt.title('PSD', fontsize=lettersize)
plt.legend(loc='best')
plt.xlabel('Freq. (Hz)', fontsize=lettersize)
plt.ylabel('Power (Index)', fontsize=lettersize)
plt.gcf().subplots_adjust(bottom=0.15)
plt.ylabel('Power (index)', fontsize=lettersize)