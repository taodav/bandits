import hashlib
from tap import Tap
from pathlib import Path
from time import time, ctime

from definitions import ROOT_DIR


class Args(Tap):
    env: str = "linear_stochastic"  # What environment do we use? (linear_stochastic)
    reward_noise: float = 0.1  # What's our noise sacle for the rewards?
    action_dim: int = 10  # What's the dimension of our bandit problem?
    lamb: int = 1  # Lambda
    total_eps: int = int(1e6)  # How many episodes do we run?

    seed: int = 2020  # Random seed for reproducibility
    log_freq: int = int(1e3)  # How often do we log?
    results_dir: Path = Path(ROOT_DIR, 'results')  # For tensorboard logging. Where do we log our files?

    def process_args(self) -> None:

        self.results_dir /= f"{self.env}_{self.action_dim}"
        self.results_dir.mkdir(parents=True, exist_ok=True)


def md5(args: Args) -> str:
    return hashlib.md5(str(args).encode('utf-8')).hexdigest()


def get_results_fname(args: Args):
    time_str = ctime(time())
    results_fname_npy = f"{md5(args)}_{time_str}.npy"
    return results_fname_npy
