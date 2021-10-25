import jax.numpy as jnp
from jax import jit


@jit
def beta(delta: float, t: float, d: int, lamb: float = 1., S: int = 1, R: int = 1):
    return R * jnp.sqrt(d * jnp.log(lamb + t) - d * jnp.log(lamb) - 2. * jnp.log(delta)) + jnp.sqrt(lamb) * S
