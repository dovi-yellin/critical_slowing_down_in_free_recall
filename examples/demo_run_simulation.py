import copy
import sys
import pickle
from datetime import datetime
import numpy as np

from csd.infrastructure.initParams import initParams, read_config
from csd.infrastructure.rate_model import RateModel
from csd.infrastructure.utils import butter_bandpass_filter

if __name__ == '__main__':
    config = read_config('test_general.json')
    random_seed = int(config['random_seed'])
    np.random.seed(random_seed)
    t_stabilize_time = int(config['params']['stabilize_time'])  # 200
    t_total_time = int(config['params']['total_time'])          # 800
    t_N = int(config['params']['N'])                            # 400
    tau = float(config['params']['tau'])                        # 20.0
    alpha = float(config['params']['alpha'])                    # 0.05
    wins = int(config['params']['wins'])                        # 24000

    # initialize simulation 1 params
    params = initParams(N=t_N, stabilize_time=t_stabilize_time, total_time=t_total_time)

    num_rows = 4
    labels = []
    results_dicts = []

    mu = params.mu
    probability = params.prb

    params.gamma = 0.081

    for i in range(num_rows):
        control_param = params.gamma * mu * probability

        # instantiate model
        model = RateModel(params=params)

        sys.stdout.flush()  # ensures progress bar starts after print
        print(f"Beginning network simulation with {params.gamma=:.6f}, {mu=}, {probability=}, {control_param=:.6f}")
        sys.stdout.flush()  # ensures progress bar starts after print

        # run the model simulation
        results_dict = model.run_local_circuit()

        # store results for later "as function of" analysis
        results_dicts.append(copy.deepcopy(results_dict))

        # Append label for plot legend
        labels.append(f"{control_param :.5f}")

        params.gamma += 0.006


b_save = True
results_dir='results'
if b_save:
    iso_8601_format = '%Y%m%dT%H%M%S'  # e.g., 20211119T221000
    fname = f"{results_dir}/rate_model_G_0.8_to_0.98.pkl"  # _run_{datetime.now().strftime(iso_8601_format)}.pkl"
    print(f"dumping results to {fname}")
    with open(fname, 'wb') as f:
        pickle.dump(results_dicts, f)
