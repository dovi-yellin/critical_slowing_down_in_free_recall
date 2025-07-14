import os
import json

# load configuration
def read_config(file_name):
    with open(file_name, "r") as f:
        return json.load(f)

class initParams(object):
    """
    initParams
    Set parameters and ensure working conditions for starting simulation

        Parameters from Chaudhuri, He and Wang 2017
        N = 440, τ = 195ms, γ = 0.1 Hz/pA, μconn = 49.881 pA/Hz, σconn = 4.988 pA/Hz, p = 0.2, and α = 1/44
    """


    def __init__(self, N=400, prb=0.20, dt=0.1, dt_record=1, stabilize_time=150, total_time=350, noise_factor_add=0.0):
        """ensure working conditions
        """
        assert stabilize_time < total_time, "total time can not be less than stabilize time"

        # Network Parameters
        self.N = N  # number of nodes
        self.dt = dt  # # time unit of simulation, interval for pace of simulation (ms)
        self.fs = (1 / dt_record) * 1000  # frequency (Hz)
        self.stabilize_time = stabilize_time  # time until network is assumed stabilized (s)

        # Time scale parameters
        self.time = total_time  # overall time to run (s)
        self.T = self.fs * self.time
        self.t_input_off = self.T

        # Model parameters
        self.alpha: 1 / 44
        self.gamma = 0.1  # f-i curve (Hz/pA)
        self.prb = prb  # probability for non-zero weight - 0.2 in paper
        self.mu = 49.881  # 0.001, # 49.881/50,  (pA/Hz)
        self.sigma = 4.988  # 4.988, (pA/Hz)
        self.tau = 195.  # time constant (ms)
        self.cap = 1000

        # Noise parameters
        self.noise_factor_add = noise_factor_add
        self.noise_factor_mult = 1.0
        self.low_pass_noise = False
        self.high_pass_noise = False

        # Block design parameters enabling change of setting within block
        self.block_size = 0
        self.gamma1 = self.gamma
        self.gamma2 = self.gamma
        self.noise_factor_add1 = noise_factor_add
        self.noise_factor_add2 = noise_factor_add

        # Prepare path and results and figures
        # todo: move to block
        if not os.path.exists('figures'):
            os.makedirs('figures')
        if not os.path.exists('results'):
            os.makedirs('results')

        # path for plots
        self.main_tag = '1'
        self.secondary_tag = '01'
        self.experiment_folder = f'Rate_model_{self.main_tag}'
        self.path = f'figures/{self.experiment_folder}/{self.secondary_tag}/'

        if not os.path.isdir(self.path):
            os.makedirs(self.path)

        # Analysis params
        # trim till start of sampling when dynamics stabilize
        self.start_sample = self.fs * self.stabilize_time

        self.dt_record = dt_record  # interval for saving record (ms)
        self.change_factor = 0.01  # a delta or multiplier factor for exploring parameter domain


    def add_to_noise_factor(self, delta) -> None:
        self.noise_factor += delta

    def __str__(self):
        return f"N={self.N}, prb={self.prb}"