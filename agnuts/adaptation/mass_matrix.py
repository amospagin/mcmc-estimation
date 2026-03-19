"""Online mass matrix adaptation via Welford's algorithm.

Estimates the posterior covariance from samples and uses it as the
mass matrix (inverse covariance = inverse mass matrix).

Supports diagonal (default) and dense estimation.
"""

import jax.numpy as jnp
from jax import Array

from agnuts.types import WelfordState


def init(dim: int, dense: bool = False) -> WelfordState:
    """Initialize Welford accumulator."""
    mean = jnp.zeros(dim)
    if dense:
        m2 = jnp.zeros((dim, dim))
    else:
        m2 = jnp.zeros(dim)
    return WelfordState(count=0, mean=mean, m2=m2)


def update(state: WelfordState, sample: Array) -> WelfordState:
    """Incorporate a new sample into the running variance estimate."""
    count = state.count + 1
    delta = sample - state.mean
    mean = state.mean + delta / count

    if state.m2.ndim == 1:
        # Diagonal: track element-wise variance
        delta2 = sample - mean
        m2 = state.m2 + delta * delta2
    else:
        # Dense: track full covariance
        delta2 = sample - mean
        m2 = state.m2 + jnp.outer(delta, delta2)

    return WelfordState(count=count, mean=mean, m2=m2)


def get_inverse_mass_matrix(
    state: WelfordState,
    regularization: float = 1e-3,
) -> Array:
    """Extract the inverse mass matrix from accumulated statistics.

    Returns the sample variance (diagonal) or covariance (dense),
    regularized for numerical stability.
    """
    if state.count < 2:
        # Not enough samples; return identity
        if state.m2.ndim == 1:
            return jnp.ones_like(state.m2)
        else:
            return jnp.eye(state.m2.shape[0])

    variance = state.m2 / (state.count - 1)

    if variance.ndim == 1:
        # Regularize: shrink toward 1.0
        return (1.0 - regularization) * variance + regularization
    else:
        # Regularize: shrink toward diagonal
        diag = jnp.diag(jnp.diag(variance))
        return (1.0 - regularization) * variance + regularization * diag
