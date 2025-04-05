""" Run the same model across a sweep of values for a single parameter.
@author: Dovi Yellin, Noam Siegel
"""

import copy
import sys

import numpy as np

from models import RateModel
from RateModelAnalysis import multi_run_analysis
from initParams import initParams, read_config


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
        val += 1 / power ** i
    arr = (arr / max(arr)) * (stop - start) + start
    return arr


if __name__ == '__main__':
    config = read_config('test_general.json') #('test_filters.json') # ('best_norman_fit_with_additive_noise_reduction.json')
    random_seed = int(config['random_seed'])
    np.random.seed(random_seed)
    t_stabilize_time = int(config['params']['stabilize_time'])  # 400
    t_total_time = int(config['params']['total_time'])          # 1400
    t_N = int(config['params']['N'])                            # 400
    tau = float(config['params']['tau'])                        # 20.0
    alpha = float(config['params']['alpha'])                    # 0.05
    wins = int(config['params']['wins'])                        # 24000
    block_size = int(config['params']['block_size'])            # 200
    noise_factor_mult = float(config['params']['noise_factor_mult'])

    # initialize simulation 1 params
    params = initParams(N=t_N, stabilize_time=t_stabilize_time, total_time=t_total_time)
    params.tau = tau
    params.noise_factor_mult = noise_factor_mult

    # instantiate model
    model = RateModel(params=params)

    probability = params.prb
    mu = params.mu
    critical_point = 1.0  # 0.9999
    delta = 0.095
    gamma = (critical_point / (probability * mu)) - delta  # 0.0983 # ext_params['gamma']
    noise_factor = 0.0

    num_models = 3
    labels = []
    results_dicts = []

    gamma_lower = (0.0000001) / (probability * mu)
    gamma_upper = (0.99) / (probability * mu) # (1 - 1e-3) / (probability * mu)

    gammas = searchspace(gamma_lower, gamma_upper, 1, 3)

    for gamma in gammas:
        control_param = gamma * mu * probability
        sys.stdout.flush()  # ensures progress bar starts after print
        print(f"Beginning network simulation with {gamma=:.6f}, {mu=}, {probability=}, {control_param=:.6f}")
        sys.stdout.flush()  # ensures progress bar starts after print

        # initialize model parameters
        model.params.gamma = gamma

        # run the model simulation
        results_dict = model.run_local_circuit()

        # Collect parameters and results
        results_dicts.append(copy.deepcopy(results_dict))

        # Append label for plot legend
        labels.append(f"{probability * gamma * mu :.5f}")

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
    multi_run_analysis(results_dicts, labels)
