import numpy as np
import jax.numpy as jnp
from jax import jit

from ban.bandits import Bandit
from ban.agents import Agent
from ban.args import Args


class Trainer:
    def __init__(self, bandit: Bandit, agent: Agent, args: Args):
        self.bandit = bandit
        self.agent = agent

        self.action_dim = args.action_dim
        self.lamb = args.lamb
        self.reward_noise = args.reward_noise
        self.total_eps = args.total_eps

        self.args = args
        self.info = {'regret': np.zeros(self.total_eps, dtype=np.float32)}
        self.log_freq = args.log_freq

    def _print_stats(self, t: int):
        print(f"Episode: {t}, \t"
              f"regret over past {self.log_freq} eps: {sum(self.info['regret'][max(t - self.log_freq, 0):t + 1]):.4f}")

    @staticmethod
    @jit
    def calc_regret(target: jnp.ndarray, theta: jnp.ndarray, action: jnp.ndarray):
        regret = target - jnp.dot(theta, action)
        return regret

    def train(self):

        for t in range(1, self.total_eps + 1):
            x_t = self.agent.act(t)

            r_t = self.bandit.pull(x_t)
            regret = self.calc_regret(1, self.agent.theta, x_t)

            self.info['regret'][t - 1] = regret

            self.agent.update(x_t, r_t)

            if t % self.log_freq == 0:
                self._print_stats(t)

    def get_info(self):
        return_info = {'args': self.args.as_dict()}
        for k, v in self.info.items():
            return_info[k] = np.array(self.info[k])

        return return_info
