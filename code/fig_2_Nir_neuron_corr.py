import pickle
import numpy as np
import pandas as pd
from scipy.signal import correlate, hilbert
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
from matplotlib import mlab
from sklearn.metrics import r2_score

from initParams import initParams
from utils import butter_bandpass_filter

#######################################################################################
#          General fitting of simulation outcome to Nir et al 2007 paper              #
#######################################################################################

# filename = 'results/fig1_subcritical_demo.pkl' # Note the current figure as of June 8, 2022 was based on this!
# filename = 'results/fig1_near_critical_demo.pkl' # 'results/fig1_subcritical_demo.pkl' # near_critical_demo.pkl' # /_2Hz.pkl'
# filename = 'results/blocks_by_gamma_modulation_run_1_seed_45.pkl'
# filename = 'results/fig2_near_critical_PSD_gamma_0.094_tau_18_seed44.pkl' # file used in fig 2 of biorxiv paper 2023
filename = "results/fig2_near_critical_PSD_gamma_0.094_tau_20_N_240_20231109.pkl"  # 'results/fig2_PSD_gamma_0.08_tau_20_N_240_20231110.pkl'
results_dicts = pickle.load(open(filename, "rb"))

if isinstance(results_dicts, list):
    results_dict = results_dicts[0]
else:
    results_dict = results_dicts
params: initParams = results_dict["params"]
N = params.N
start_sample = 250000  # int(params.start_sample)
dt = params.dt
fs = params.fs
total_time = params.time
fs_raw = fs / dt
T = total_time * fs
n_t = int(round(T // dt)) + 1

r_store = results_dict["r_store"]
activity_per_unit = np.transpose(r_store)
node_activities = activity_per_unit[:, (start_sample - 1) : -1]
node_activities = (
    node_activities - node_activities.mean(axis=1, keepdims=True)
) / node_activities.std(axis=1, keepdims=True)
node_activities_filt = butter_bandpass_filter(node_activities, 0.00001, 0.1, fs)
node_activities_filt = (
    node_activities_filt - node_activities_filt.mean(axis=1, keepdims=True)
) / node_activities_filt.std(axis=1, keepdims=True)

sig_len = len(node_activities_filt[0, :])
xvec = np.linspace(0, int(sig_len / fs), sig_len)

# consider looking at the hilbert profile of the activity rate...
analytical_signal = hilbert(node_activities_filt)
amplitude_envelope = np.abs(analytical_signal)

b_corr_on_all_pairs = True
if b_corr_on_all_pairs:
    for i in range(0, N - 1):
        for j in range(i + 1, N):
            corr, _ = pearsonr(node_activities_filt[i, :], node_activities_filt[j, :])
            if corr > 0.5:
                print(f"pair of neurons are at indexes {i}, {j}, corr = {corr}")
                break

# plot activity per unit over first 2 units as needed in figure 1
lettersize = 24
fig = plt.figure(1, figsize=(20, 4))
plt.plot(xvec, node_activities_filt[0, :], linewidth=3, color="red")
plt.plot(xvec, node_activities_filt[1, :], linewidth=3, color="blue")
# option to present activity rate of many more neurons (with or without filtering)
# for i in range(0, N):
#     # plt.plot(xvec, node_activities[i, :])
#     plt.plot(xvec, node_activities_filt[i, :])
#     # plt.plot(xvec, amplitude_envelope[i, :])
plt.xlabel("Time (sec)", fontsize=lettersize)
plt.gcf().subplots_adjust(bottom=0.15)
plt.ylabel("Activity rate (Z Score)", fontsize=lettersize)
plt.yticks(np.arange(-4, 4, 2))
plt.yticks(fontsize=18)
plt.ylim(-3.5, 3.5)
plt.xlim(100, 300)
plt.gca().spines["right"].set_color("none")
plt.gca().spines["top"].set_color("none")
plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
fig.tight_layout()

corr, _ = pearsonr(node_activities_filt[0, :], node_activities_filt[1, :])

# compute auto- and cross-correlation for selection of neurons and pairs
auto_corr_arr = []
cross_corr_arr = []
size = node_activities[0].size
for i in range(1, 40, 2):
    ac = (
        correlate(
            node_activities[i, :], node_activities[i, :], mode="same", method="auto"
        )
        / size
    )
    auto_corr_arr.append(ac)
    cc = (
        correlate(
            node_activities[i, :], node_activities[i + 1, :], mode="same", method="auto"
        )
        / size
    )
    cross_corr_arr.append(cc)

auto_corr = np.mean(auto_corr_arr, axis=0)
cross_corr = np.mean(cross_corr_arr, axis=0)

fig = plt.figure(2, figsize=(4, 6))
plt.plot(xvec, auto_corr, linewidth=3, color="black")
plt.plot(xvec, cross_corr, linewidth=3, color="orange")


size = auto_corr.size
win = 20000
indices = np.linspace(size / 2 - win, size / 2 + win, win * 2 + 1).astype(int)

# s1 = np.abs(np.fft.rfft(auto_corr))
# fr1 = np.fft.rfftfreq(auto_corr.size, d=1./fs)
# s2 = np.abs(np.fft.rfft(cross_corr))
# fr2 = np.fft.rfftfreq(cross_corr.size, d=1./fs)
seg = 1
s1, fr1 = mlab.psd(auto_corr[indices], NFFT=int(auto_corr[indices].size / seg), Fs=fs)
s2, fr2 = mlab.psd(cross_corr[indices], NFFT=int(cross_corr[indices].size / seg), Fs=fs)
k = 200000  # scaling factor for Welch
s1 *= seg * k
s2 *= seg * k

# fig = plt.figure(3, figsize=(8, 7))
fig, ax = plt.subplots(figsize=(8, 7))
plt.loglog(fr1, s1, linewidth=3, color="black")  # auto_corr in black
plt.loglog(fr2, s2, linewidth=3, color="orange")  # cross_corr in orange
# plt.grid()
plt.xlabel("Freq. (Hz)", fontsize=lettersize)
plt.ylabel("Power (index)", fontsize=lettersize)
plt.yticks(fontsize=24)
plt.xticks(fontsize=24)
# plt.xticks(fr, [ f"{int(np.log10(x))}" for x in fr])
plt.ylim(0.6, 700)
plt.xlim(0.01, 12)
plt.gca().spines["right"].set_color("none")
plt.gca().spines["top"].set_color("none")
plt.gcf().subplots_adjust(bottom=0.15, left=0.15)
# plt.yticks(np.arange(0, 2, 1))

ind = [(fr1 >= 0.01) & (fr1 <= 100)]
fr1 = fr1[ind]
s1 = s1[ind]
fr2 = fr2[ind]
s2 = s2[ind]


# fitting an exponential function that looks like a line on a lof-log plot.
def myExpFunc(x, a, b):
    return a * np.power(x, b)


popt1, pcov1 = curve_fit(myExpFunc, fr1, s1)
popt2, pcov2 = curve_fit(myExpFunc, fr2, s2)

newX = np.logspace(-1, 1, base=10)
newY1 = myExpFunc(fr1, *popt1)
newY2 = myExpFunc(fr2, *popt2)
plt.plot(fr1, newY1, "k")
plt.plot(fr2, newY2, "k")
fig.tight_layout()


r2_1 = r2_score(s1, newY1)
r2_2 = r2_score(s2, newY2)
a1 = popt1[1]
b1 = np.log10(popt1[0])
a2 = popt2[1]
b2 = np.log10(popt2[0])

ax.text(0.015, 150, f"y = {a1}x + {b1}", fontsize=20, color="black")
ax.text(0.015, 50, f"y = {a2}x + {b2}", fontsize=20, color="black")
# from scipy.stats import linregress
# slope, intercept, r_value, p_value, std_err = linregress(np.log10(fr1+1), np.log10(s1))

# Create sequence of 100 numbers and draw line
# xseq = np.linspace(0, 10, num=100)
# xseq = np.log10(np.logspace(-2, 2, base=10))
# plt.plot(xseq, xseq*slope+intercept)

# PSD for figure 1 in imitation of Nir
alpha = 0.1
mean_activity = np.mean(node_activities_filt[0 : int(alpha * params.N), :], axis=0)
mean_activity = mean_activity - np.mean(mean_activity)

# plot the Welch PSD for mean activity
s, fr = mlab.psd(mean_activity, NFFT=int(mean_activity.size / 100), Fs=fs)
fig = plt.figure(figsize=(6, 5))
plt.loglog(fr, s)
plt.grid()
plt.xlabel("Freq. (Hz)", fontsize=lettersize)
plt.ylabel("Power (index)", fontsize=lettersize)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
# plt.xticks(fr, [ f"{int(np.log10(x))}" for x in fr])
plt.gcf().subplots_adjust(bottom=0.15, left=0.15)
# plt.ylim(0.000001, 100000)
fig.tight_layout()


# load dataframe with summary of correlational analysis
df_corr = pd.read_csv(
    "C:/Research/Spontaneous_activity/Rate_model/CriticalSlowDown/research-critical-slow-down/data/fig2_correlations.csv"
)

lettersize = 24
x_gamma = df_corr.values[:, 0] * 0.2 * 49.881
fig = plt.figure(figsize=(8, 8))
plt.plot(
    x_gamma, df_corr.values[:, 1], label="Max correlation", linewidth=4, color="blue"
)  # max neuroal correlation
plt.plot(
    x_gamma, df_corr.values[:, 2], label="Mean correlation", linewidth=4, color="green"
)  # mean neuronal correlation

# plt.plot(x_gamma, np.divide(np.log(df_corr.values[:,3]),np.log(df_corr.values[:,4])), label='Auto-to-cross ratio', linewidth=4, color='orange')
plt.axhline(y=0.56, color="r", linestyle="-")
plt.legend(loc="best", fontsize=lettersize)
plt.xlabel("Control parameter", fontsize=lettersize)
plt.ylabel("Correlation", fontsize=lettersize)
plt.yticks(fontsize=lettersize)
plt.xticks([0.8, 0.85, 0.9, 0.95, 1.0], fontsize=lettersize)
fig.tight_layout()

# two sub-plot alternative
fig = plt.figure()

ax1 = plt.subplot(211)
ax1.plot(
    x_gamma, df_corr.values[:, 1], label="Max correlation", linewidth=4, color="blue"
)  # max neuroal correlation
ax1.plot(
    x_gamma, df_corr.values[:, 2], label="Mean correlation", linewidth=4, color="green"
)  # mean neuronal correlation
ax1.axhline(y=0.56, color="r", linestyle="-")
ax1.legend(loc="best", fontsize=lettersize)
# ax1.set(ylabel="Correlation") #ax1.ylabel('Correlation', fontsize=lettersize)
ax1.set_ylabel("Correlation", fontsize=lettersize)
plt.yticks(fontsize=lettersize)

ax2 = plt.subplot(212)
ax2.plot(
    x_gamma,
    np.divide(np.log(df_corr.values[:, 3]), np.log(df_corr.values[:, 4])),
    label="Auto-to-cross corr ratio",
    linewidth=4,
    color="orange",
)
ax2.legend(loc="best", fontsize=lettersize)
ax2.axhline(y=0.84, color="r", linestyle="-")
ax2.set_ylabel("Ratio", fontsize=lettersize)  # ax2.set(ylabel="Ratio")

ax1.get_shared_x_axes().join(ax1, ax2)
ax1.set_xticklabels([])
# ax2.autoscale() ## call autoscale if needed

plt.yticks(fontsize=lettersize)
plt.xlabel("Control parameter", fontsize=lettersize)
plt.xticks([0.8, 0.85, 0.9, 0.95, 1.0], fontsize=lettersize)
# plt.ylabel(fontsize=lettersize)
fig.tight_layout()

# load dataframe with summary of correlation by network size analysis
df_corr = pd.read_csv(
    "C:/Research/Spontaneous_activity/Rate_model/CriticalSlowDown/research-critical-slow-down/data/fig_SI_network_correlation_by_N.csv"
)
df_corr_2 = pd.read_csv(
    "C:/Research/Spontaneous_activity/Rate_model/CriticalSlowDown/research-critical-slow-down/data/fig_SI_network_correlation_by_N_2.csv"
)

df_corr = df_corr + df_corr_2
df_corr = df_corr / 2

lettersize = 24
N = df_corr.values[:, 0]
fig = plt.figure(figsize=(8, 8))
# ax2 = ax.twinx()
plt.plot(
    N, df_corr.values[:, 2], label="Max correlation", linewidth=4, color="blue"
)  # max neuroal correlation
plt.plot(N, df_corr.values[:, 3], label="Mean correlation", linewidth=4, color="green")

plt.plot(
    N, df_corr.values[:, 6], label="Normalized sum power", linewidth=4, color="orange"
)
# ax2.set_ylabel('Power', color='orange')
# plt.plot(N, np.divide(df_corr.values[:,5]),np.log(df_corr.values[:,4]), label='Auto-to-cross corr ratio', linewidth=4, color='orange')

plt.legend(loc="best", fontsize=lettersize)
plt.xlabel("N", fontsize=lettersize)
plt.ylabel("Correlation", fontsize=lettersize)
plt.xticks(fontsize=lettersize)
plt.yticks(fontsize=lettersize)

# example for secondary y-axis, in case all data should fit in single plot
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(
    N, df_corr.values[:, 2], label="Max correlation", linewidth=4, color="blue"
)  # max neuroal correlation
ax1.plot(N, df_corr.values[:, 3], label="Mean correlation", linewidth=4, color="green")
plt.yticks(fontsize=lettersize)

ax2.plot(
    N, df_corr.values[:, 6], label="Normalized sum power", linewidth=4, color="orange"
)
plt.yticks(fontsize=lettersize)
plt.xticks(fontsize=lettersize)
ax1.set_xlabel("N", fontsize=lettersize)
ax1.set_ylabel("Correlation", fontsize=lettersize)
ax2.set_ylabel("Normalized sum power", color="orange", fontsize=lettersize)
ax1.legend(loc="best", fontsize=lettersize)
ax2.legend(loc="best", fontsize=lettersize)
