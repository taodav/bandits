import numpy as np
import jax.numpy as jnp
from jax import jit, random
from typing import Any


class Bandit:
    def pull(self, action: Any):
        raise NotImplementedError


class LSBandit(Bandit):
    def __init__(self, dim: int, rand_key: random.PRNGKey, noise_scale: float = 0.1):
        self.dim = dim

        self.noise_scale = jnp.array(noise_scale, dtype=float)
        self.theta = np.zeros(self.dim, dtype=float)
        self.theta[0] = 1.
        self.rand_key = rand_key

    @staticmethod
    @jit
    def functional_pull(theta: jnp.ndarray, action: jnp.ndarray, noise_scale: jnp.ndarray, rand_key: random.PRNGKey):
        key, subkey = random.split(rand_key)
        noise = random.normal(subkey, shape=(1,))*noise_scale
        rew = jnp.dot(theta, action)
        return rew + noise, key

    def pull(self, action: np.ndarray):
        r_t, self.rand_key = self.functional_pull(self.theta, action, self.noise_scale, self.rand_key)
        return r_t
