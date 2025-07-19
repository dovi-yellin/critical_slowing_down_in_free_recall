import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import mlab

from csd.infrastructure.utils import butter_lowpass_filter
from csd.infrastructure.matlab_to_numpy import loadmat

fs = 1000

# fetch Norman et al., 2017 data
fpath = r"..\data\figdata.mat"
d = loadmat(fpath)

X = np.asarray(d["figdata"]["rawSpectrum"]["X"])
Y = np.asarray(d["figdata"]["rawSpectrum"]["Y"])

fig = plt.figure(1, figsize=(10, 10))
lettersize = 24
plt.semilogx(
    X[0], Y[0], label="Norman - Recall", linewidth=6, alpha=0.9, color="red"
)  # plt.plot(X[0], Y[0])
plt.semilogx(
    X[1], Y[1], label="Norman - Resting state", linewidth=6, alpha=0.9, color="blue"
)
plt.xlabel("Freq. (Hz)", fontsize=lettersize * 1.5)
plt.gcf().subplots_adjust(bottom=0.15)
plt.ylabel("Power (dB)", fontsize=lettersize * 1.5)
plt.legend(loc="lower left", fontsize=20)
plt.xlim(0.025, 18)
plt.ylim(-20, 2)
plt.xticks(fontsize=lettersize)
plt.yticks(fontsize=lettersize)
fig.tight_layout()

###################################################################################
# Load simulation results as stored in pkl files (due to their large size, these files were not uploaded to github)
# Use JSON configuration file to simulate and reconstruct files marked in comment below
###################################################################################

filenames_additive_noise = [
    "results/fig3/blocks_by_additive_noise_run_1_seed_40_additive_12.5.pkl",
    "results/fig3/blocks_by_additive_noise_run_1_seed_41_additive_12.5.pkl",
    "results/fig3/blocks_by_additive_noise_run_1_seed_42_additive_12.5.pkl",
    "results/fig3/blocks_by_additive_noise_run_1_seed_44_additive_12.5.pkl",
    "results/fig3/blocks_by_additive_noise_run_1_seed_45_additive_12.5.pkl",
    "results/fig3/blocks_by_additive_noise_run_1_seed_47_additive_12.5.pkl",
    "results/fig3/blocks_by_additive_noise_run_1_seed_48_additive_12.5.pkl",
    "results/fig3/blocks_by_additive_noise_run_1_seed_49_additive_12.5.pkl",
]

nfft = 9000
K = 700  # scaling factor to match arbitrary metric in Norman et al power index
spct_blocks1 = []
spct_blocks2 = []
for filename in filenames_additive_noise:
    result_dict = pickle.load(open(filename, "rb"))

    # Compute blocks' PSD
    activity_block1 = result_dict["activity_rest"]
    mean_reduced_block1 = activity_block1 - np.mean(activity_block1)

    activity_block2 = result_dict["activity_CSD"]
    mean_reduced_block2 = activity_block2 - np.mean(activity_block2)

    cutoff = 15.0
    mean_block1_lp = butter_lowpass_filter(mean_reduced_block1, cutoff, fs)
    mean_block2_lp = butter_lowpass_filter(mean_reduced_block2, cutoff, fs)

    spct, freqs1 = mlab.psd(mean_block1_lp, NFFT=nfft, Fs=fs)
    spct = spct * K
    spct_db1 = 10 * np.log10(spct)

    spct, freqs2 = mlab.psd(mean_block2_lp, NFFT=nfft, Fs=fs)
    spct = spct * K
    spct_db2 = 10 * np.log10(spct)

    spct_blocks1.append(spct_db1)
    spct_blocks2.append(spct_db2)

mean_spct_block1 = np.mean(spct_blocks1, axis=0)
var_spct_block1 = np.std(spct_blocks1, axis=0)  # stats.sem(spct_blocks1, axis=0)

mean_spct_block2 = np.mean(spct_blocks2, axis=0)
var_spct_block2 = np.std(spct_blocks2, axis=0)  # stats.sem(spct_blocks2, axis=0)

fig = plt.figure(1, figsize=(10, 10))
lettersize = 28
plt.semilogx(
    freqs1,
    mean_spct_block1,
    label="Simulation - rest state",
    linewidth=4,
    color="green",
)
plt.fill_between(
    freqs1,
    mean_spct_block1 - var_spct_block1,
    mean_spct_block1 + var_spct_block1,
    color="green",
    alpha=0.4,
)
plt.semilogx(
    freqs2, mean_spct_block2, label="Simulation - recall", linewidth=4, color="orange"
)
plt.fill_between(
    freqs2,
    mean_spct_block2 - var_spct_block2,
    mean_spct_block2 + var_spct_block2,
    color="orange",
    alpha=0.4,
)
plt.semilogx(
    X[0], Y[0], label="Norman - Recall", linewidth=2, alpha=0.9, color="red"
)
plt.semilogx(
    X[1], Y[1], label="Norman - Resting state", linewidth=2, alpha=0.9, color="blue"
)

plt.xlabel("Freq. (Hz)", fontsize=lettersize * 1.5)
plt.gcf().subplots_adjust(bottom=0.15)
plt.ylabel("Power (dB)", fontsize=lettersize * 1.5)
plt.legend(loc="best", fontsize=lettersize)
plt.xlim(0.025, 18)
plt.ylim(-20, -5)
plt.xticks(fontsize=lettersize)
plt.yticks(fontsize=lettersize)
fig.tight_layout()
plt.close()

# compare between blocks
# mean_activity = np.mean(model_activity[0:int(alpha * N) ,:], axis=0)
# mean_activity = mean_activity - np.mean(mean_activity)

# perform statistical analysis on simulation results (relative to empirical)
low_fr_cutoff = 0.5
range_spect1_range_low = np.sum(
    np.asarray(spct_blocks1)[:, np.logical_and(freqs1 > 0, freqs1 <= low_fr_cutoff)],
    axis=1,
)
range_spect2_range_low = np.sum(
    np.asarray(spct_blocks2)[:, np.logical_and(freqs1 > 0, freqs1 <= low_fr_cutoff)],
    axis=1,
)
diff_spect_range_low = (range_spect2_range_low - range_spect1_range_low) / sum(
    freqs1 < low_fr_cutoff
)

diff_spect_by_freq = np.asarray(spct_blocks2) - np.asarray(spct_blocks1)

mu = 0

t_value, p_value = stats.ttest_1samp(diff_spect_by_freq[:, 3], mu)

one_tailed_p_value = float(
    "{:.6f}".format(p_value / 2)
)  # Since alternative hypothesis is one tailed - need to divide p value by 2.
print("Test statistic is %f" % float("{:.6f}".format(t_value)))
print("p-value for one tailed test is %f" % one_tailed_p_value)

w, p = stats.wilcoxon(diff_spect_by_freq[:, 1])

# for Fig 5c - show how additional noise influences the CSD
filenames_additive_noise = [
    "results/fig3/blocks_by_additive_noise_run_1_seed_45_additive_10.pkl",
    "results/fig3/blocks_by_additive_noise_run_1_seed_45_additive_15.pkl",
    "results/fig3/blocks_by_additive_noise_run_1_seed_45_additive_25.pkl",
    "results/fig3/blocks_by_additive_noise_run_1_seed_45_additive_50.pkl",
]

nfft = 10000
spct_blocks1 = []
spct_blocks2 = []
for filename in filenames_additive_noise:
    result_dict = pickle.load(open(filename, "rb"))

    # Compute blocks' PSD
    # ex = int(fs * 10) # remove additional few seconds at start of blocks, till signal stabilizes
    activity_block1 = result_dict["activity_rest"]
    mean_reduced_block1 = activity_block1 - np.mean(activity_block1)

    activity_block2 = result_dict["activity_CSD"]
    mean_reduced_block2 = activity_block2 - np.mean(activity_block2)

    cutoff = 15.0
    mean_block1_lp = butter_lowpass_filter(mean_reduced_block1, cutoff, fs)
    mean_block2_lp = butter_lowpass_filter(mean_reduced_block2, cutoff, fs)

    spct, freqs1 = mlab.psd(mean_block1_lp, NFFT=nfft, Fs=fs)
    spct = spct * K
    spct_db1 = 10 * np.log10(spct)

    spct, freqs2 = mlab.psd(mean_block2_lp, NFFT=nfft, Fs=fs)
    spct = spct * K
    spct_db2 = 10 * np.log10(spct)

    spct_blocks1.append(spct_db1)
    spct_blocks2.append(spct_db2)

fig = plt.figure(1, figsize=(10, 10))
lettersize = 24
plt.semilogx(
    freqs1, spct_blocks1[0], label="Simulation - rest state", linewidth=2, color="green"
)  # plt.plot(freqs, spct_db)
plt.semilogx(
    freqs2,
    spct_blocks2[0],
    label="Simulation - recall - additive noise 10",
    linewidth=2,
    color=[1.0, 0.9, 0.25],
)
plt.semilogx(
    freqs2,
    spct_blocks2[1],
    label="Simulation - recall - additive noise 15",
    linewidth=2,
    color=[1.0, 0.85, 0.25],
)
plt.semilogx(
    freqs2,
    spct_blocks2[2],
    label="Simulation - recall - additive noise 25",
    linewidth=2,
    color=[1.0, 0.8, 0.25],
)
plt.semilogx(
    freqs2,
    spct_blocks2[3],
    label="Simulation - recall - additive noise 50",
    linewidth=2,
    color=[1.0, 0.75, 0.25],
)
plt.semilogx(
    X[0], Y[0], label="Norman - Recall", linewidth=6, alpha=0.9, color="red"
)  # plt.plot(X[0], Y[0])
plt.semilogx(
    X[1], Y[1], label="Norman - Resting state", linewidth=6, alpha=0.9, color="blue"
)
plt.xlabel("Freq. (Hz)", fontsize=lettersize * 1.5)
plt.gcf().subplots_adjust(bottom=0.15)
plt.ylabel("Power (dB)", fontsize=lettersize * 1.5)
plt.legend(loc="lower left", fontsize=20)
plt.xlim(0.025, 18)
plt.ylim(-20, 2)
plt.xticks(fontsize=lettersize)
plt.yticks(fontsize=lettersize)
fig.tight_layout()
