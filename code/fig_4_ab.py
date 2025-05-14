"""Run the same model across a sweep of values for a single parameter.
@author: Dovi Yellin, Noam Siegel
"""

import copy
import sys

import numpy as np

from models import RateModel
from initParams import initParams
import pickle
from datetime import datetime
import matplotlib.pyplot as plt


def searchspace(start, stop, power, num):
    """
    Create a search space scaled between `start` and `stop` with progressively smaller increments (follow 1/(2^n))
    Implemented as partial sums of geometric series with a negative exponent (e.g., arr[4] = 1 + 1/2 + 1/4 + 1/8))
    Example: searchspace(0.95, 0.999, 2, 5) -> [0.95, 0.97613333, 0.9892, 0.99573333, 0.999]
    Returns: array of search space elements
    """
    arr = np.zeros(num)
    val = 0
    for i in range(1, num + 1):
        arr[i - 1] = val
        val += 1 / power**i
    arr = (arr / max(arr)) * (stop - start) + start
    return arr


def welch_spectrum_plot(results_dict, fig, lettersize, label, scale=1.0):
    # retrieve and plot rate signal
    activity_per_unit = np.transpose(results_dict["r_store"])
    params = results_dict["params"]
    model_activity = activity_per_unit[:, (int(params.start_sample) - 1) : -1]
    # mean_activity = np.mean(model_activity[0:int(1.0 * params.N), :], axis=0)
    # mean_activity = mean_activity - np.mean(mean_activity)
    mean_activity = np.mean(model_activity[0 : int(1.0 * params.N), :], axis=1)
    mean_activity = mean_activity[:, np.newaxis]
    mean_reduced_activity = model_activity - mean_activity
    std_activity = np.std(model_activity[0 : int(1.0 * params.N), :], axis=1)
    z_score_activity = np.divide(mean_reduced_activity, std_activity[:, None])
    mean_zactivity = np.mean(z_score_activity[0 : int(1.0 * params.N), :], axis=0)

    # plot the Welch PSD for figure 4
    from matplotlib import mlab

    s, fr = mlab.psd(mean_zactivity, NFFT=20000, Fs=params.fs)
    s = s * 1.0  # scale
    plt.loglog(fr, s, label=label)


if __name__ == "__main__":
    # numpy seed for reproducibility
    np.random.seed(44)

    # initialize simulation params
    params = initParams(stabilize_time=100, total_time=300)

    # instantiate model
    model = RateModel(params=params)

    # todo: check if we can remove this?
    mu_W = []
    sigma_W = []

    probability = params.prb
    mu = params.mu
    critical_point = 1.0  # 0.9999
    delta = 0.095
    gamma = (
        critical_point / (probability * mu)
    ) - delta  # 0.0983 # ext_params['gamma']
    noise_factor = 0.0

    num_models = 5
    labels = []
    results_dicts = []

    gamma_lower = (0.99) / (probability * mu)
    gamma_upper = (0.9999) / (probability * mu)  # (1 - 1e-3) / (probability * mu)

    # gammas = searchspace(gamma_lower, gamma_upper, 1, num_models)
    def calculate_gamma(G):
        return (G) / (probability * mu)

    gammas = np.array(
        [
            calculate_gamma(0.0000001),
            calculate_gamma(0.5),
            calculate_gamma(0.947),
            calculate_gamma(0.99),
            calculate_gamma(0.999),
        ]
    )

    for gamma in gammas:
        control_param = gamma * mu * probability
        sys.stdout.flush()  # ensures progress bar starts after print
        print(
            f"Beginning network simulation with {gamma=:.6f}, {mu=}, {probability=}, {control_param=:.6f}"
        )
        sys.stdout.flush()  # ensures progress bar starts after print

        # initialize model parameters
        model.params.gamma = gamma

        # run the model simulation
        results_dict = model.run_local_circuit()

        # Collect parameters and results
        results_dicts.append(copy.deepcopy(results_dict))

        # Append label for plot legend
        labels.append(f"{probability * gamma * mu:.5f}")

        # mu_W.append(np.mean(model.W))
        # sigma_W.append(np.var(model.W))

        # optionally - modulate connections
        # model.modulate_connections_by_addition(ext_params['change_factor'])
        # model.modulate_connections_by_multiplier(ext_params['change_factor']) # (1.1)
        # model.add_to_noise_factor(0.1)
        # model.modulate_noise_by_multiplier(2)

        # gamma += delta/9 # 0.00033
        # labels.append(str(noise_factor_add))
        # noise_factor_add += 5

    # Analyze simulation results - plot circuit activity profiles
    # multi_run_analysis(results_dicts, labels)

    fig = plt.figure(1, figsize=(10, 10))
    lettersize = 32
    plt.grid()
    plt.xlabel("Freq. (Hz)", fontsize=lettersize)
    plt.ylabel("Power (index)", fontsize=lettersize)
    for idx, results_dict in enumerate(results_dicts):
        print(f"running analysis of model {idx + 1}/{len(results_dicts)}")
        params = results_dict["params"]
        print(f"params ares {params}")

        scale = 1.0
        if idx == 0:
            scale = 100000000000000
        if idx == 1:
            scale = 2
        welch_spectrum_plot(results_dict, fig, lettersize, labels[idx])

    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24)
    plt.legend(loc="best", fontsize=18)
    # plt.xticks(fr, [ f"{int(np.log10(x))}" for x in fr])
    plt.gcf().subplots_adjust(bottom=0.15, left=0.15)
    # plt.ylim(0.000001, 100000)
    fig.tight_layout()


b_save = True
results_dir = "results"
if b_save:
    iso_8601_format = "%Y%m%dT%H%M%S"  # e.g., 20211119T221000
    fname = f"{results_dir}/fig_4a_run_{datetime.now().strftime(iso_8601_format)}.pkl"
    print(f"dumping results to {fname}")
    with open(fname, "wb") as f:
        pickle.dump(results_dicts, f)
