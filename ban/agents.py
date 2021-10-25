import numpy as np
import jax.numpy as jnp
from jax import jit, random
from functools import partial

from ban.utils import beta


class Agent:
    def act(self, *args):
        raise NotImplementedError

    def update(self, *args):
        raise NotImplementedError


class LinearThomsonAgent(Agent):
    def __init__(self, action_dim: int, lamb: int, rand_key: random.PRNGKey,
                 S: int = 1, R: int = 1):
        self.action_dim = jnp.array(action_dim)
        self.lamb = jnp.array(lamb)
        self.S = jnp.array(S)
        self.R = jnp.array(R)

        self.theta = np.zeros(action_dim)
        self.rsxs = np.zeros_like(self.theta)
        self.cov = np.identity(self.action_dim) * lamb
        self.means = jnp.zeros(self.action_dim)
        self.inv = self.cov
        self.delta = jnp.array(1 / self.action_dim)
        self.rand_key = rand_key

    @staticmethod
    @jit
    def functional_act(theta, t: jnp.ndarray, means: jnp.ndarray, inv: jnp.ndarray, delta: jnp.ndarray,
                       action_dim: jnp.ndarray, lamb: jnp.ndarray,
                       S: jnp.ndarray, R: jnp.ndarray,
                       rand_key: random.PRNGKey):

        key, subkey = random.split(rand_key)
        b = beta(delta, t, action_dim, lamb, S, R)

        theta_hat = theta + b * random.multivariate_normal(subkey, means, inv)

        # Argmax over arms
        x_t = theta_hat / jnp.linalg.norm(theta_hat, ord=2)

        return x_t, key

    def act(self, t: int):
        action, self.rand_key = self.functional_act(self.theta, t, self.means, self.inv, self.delta, self.action_dim, self.lamb, self.S, self.R, self.rand_key)
        return action

    @partial(jit, static_argnums=0)
    def functional_update(self, x_t: np.ndarray, r_t: float, cov: np.ndarray, rsxs: np.ndarray):
        cov += jnp.outer(x_t, x_t)
        rsxs += x_t * r_t

        inv = jnp.linalg.inv(cov)
        theta = jnp.matmul(inv, rsxs)
        return cov, rsxs, inv, theta

    def update(self, x_t: np.ndarray, r_t: float):
        self.cov, self.rsxs, self.inv, self.theta = self.functional_update(x_t, r_t, self.cov, self.rsxs)