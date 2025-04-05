# -*- coding: utf-8 -*-
"""Rate model.
Implementation of the Rate model was based on github submission of:
Brain-wide Maps Reveal Stereotyped Cell-Type-Based Cortical Architecture and Subcortical Sexual Dimorphism
Cell 2017
@author: Guangyu Robert Yang, 2015-2017"""

import pickle
from datetime import datetime

import numpy as np
from numpy import ndarray
from tqdm import tqdm
from scipy.stats import norm
from initParams import initParams
from utils import butter_lowpass_filter, butter_highpass_filter

weights = ndarray  # type alias

class RateModel(object):
    """The model."""

    def __init__(self, params: initParams):
        """
        params: (initParams) | all the simulation and model parameters)
        """
        self.rng = np.random.RandomState(520)

        def relu(x):
            return x * (x > 0)

        self.relu = relu

        self.params = params
        # Initialize network weights
        N_nonzero = int(params.N * params.N * params.prb)
        N_zero = (params.N * (params.N-1)) - N_nonzero # number of zeros needed taking non-zero and diagonal elements into account

        W_nonzero = np.random.normal(loc=params.mu, scale=params.sigma, size=N_nonzero) / params.N
        W_zero = np.zeros(N_zero)
        W_combined = np.concatenate([W_zero, W_nonzero])
        np.random.shuffle(W_combined)
        for i in range(params.N):  # add diagonal zeros
            W_combined = np.insert(W_combined, [i * params.N + i], [0])
        W_combined = W_combined.reshape(params.N, params.N)
        # np.fill_diagonal(W_combined, 0)
        self.W: weights = W_combined # .reshape(params.N, params.N)

    @property
    def W_eff(self) -> ndarray:
        """Future - Calculate the local effective connectivity (network coupling) matrix."""
        params = self.params
        # Effective weight matrix
        W_eff = ((self.W - np.eye(params.N)).T / params.tau).T  # noam: missing gamma? SI Section 4 Eq. 5
        return W_eff

    def modulate_connections_by_addition(self, delta) -> None:
        self.W[self.W > 0] += delta  # self.W = self.W + delta
        self.W[self.W < 0] -= delta
        return

    def modulate_connections_by_multiplier(self, mult) -> None:
        self.W = self.W * mult
        return

    def modulate_noise_by_multiplier(self, factor) -> None:
        self.noise_factor_mult *= factor
        self.p['noise_factor_mult'] = self.noise_factor_mult
        return

    def replace_connections(self, W) -> None:
        self.W = W
        return

    def run_local_circuit(self, results_dir='results', b_save=True):
        """
        Run local circuit

        Args:
            results_dir: relative path to results directory
            b_save: whether to save results to disk

        Returns: None

        """

        params = self.params
        # Simulation parameters
        N = params.N
        fs = params.fs
        dt = params.dt
        fs_raw = int(fs / dt)
        dt_record = params.dt_record
        T = params.T
        n_t = int(round(T // dt)) + 1
        n_recorddt = int(round(dt_record / dt))
        gamma = params.gamma
        tau = params.tau
        noise_factor_add = params.noise_factor_add
        noise_factor_mult = params.noise_factor_mult

        cap = params.cap
        t_input_off = params.t_input_off
        stabilize_time = params.stabilize_time
        block_size = params.block_size

        # Initialize activity to background firing
        r = np.zeros(N)  # np.random.random(N) # r_tgt

        noise_fitered_arr = []
        cutoff = 10
        if params.low_pass_noise or params.high_pass_noise:
            white_noise_arr = np.random.random((N, n_t))
            if params.low_pass_noise:
                noise_fitered_arr = butter_lowpass_filter(white_noise_arr, cutoff, fs_raw)
            if params.high_pass_noise:
                noise_fitered_arr = butter_highpass_filter(white_noise_arr, cutoff, fs_raw)

        # Storage
        r_store = []
        t_plot = []
        ext_input = []
        I_stim_store = []

        # future option for simulating external stimuli
        I_stim = np.zeros(N)

        # Running the network
        block_counter = 0
        pbar = tqdm(range(n_t), desc="running the network")  # a tiny progress bar
        for i_t in pbar:
            t = i_t * dt
            i_T = t / fs

            if block_size > 0 and i_T >= stabilize_time:
                if i_T % block_size == 0:
                    gamma = params.gamma1
                    noise_factor_add = params.noise_factor_add1

                    if block_counter % 2 == 1:
                        gamma = params.gamma2
                        noise_factor_add = params.noise_factor_add2

                    block_counter += 1

            # add background random noise to k random nodes
            I_bkg = np.zeros(N)
            if t < t_input_off:

                k = int(N)  # k: number of nodes which get noise

                if len(noise_fitered_arr) > 0:
                    noise_arr = noise_fitered_arr[:, i_t]
                else:
                    noise_arr = np.random.random(k) # white noise
                I_bkg[:k] = (noise_arr * noise_factor_mult) + noise_factor_add
                np.random.shuffle(I_bkg)

            # SI Section 4 Eq. 1
            # get internal recurrent currents
            I_local = np.dot(self.W, r)
            # evolve firing rates 1 dt time step.
            r = r + (-r + self.relu(I_local + I_stim + I_bkg) * gamma) * dt / tau

            # clip firing rates r to be between 0 and cap
            r = np.clip(r, 0, cap)

            # store values for analysis
            if i_t % n_recorddt == 0:
                r_store.append(r)
                ext_input.append(I_bkg.copy())
                t_plot.append(t)
                # I_stim_store.append(I_stim)

        result = {'r_store': np.array(r_store),
                  # 'I_stim': np.array(I_stim_store),
                  'ext_input_store': np.array(ext_input),
                  'params': params,
                  't_plot': np.array(t_plot),
                  'type': 'local_run',
                  'W_eff': self.W_eff}

        b_save = False
        if b_save:
            iso_8601_format = '%Y%m%dT%H%M%S'  # e.g., 20211119T221000
            fname = f"{results_dir}/run_{datetime.now().strftime(iso_8601_format)}.pkl"
            print(f"dumping results to {fname}")
            with open(fname, 'wb') as f:
                pickle.dump(result, f)

        return result
