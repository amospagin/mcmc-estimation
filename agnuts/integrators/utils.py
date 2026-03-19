"""Utility functions for Hamiltonian dynamics."""

import jax
import jax.numpy as jnp
from jax import Array


def kinetic_energy(momentum: Array, inverse_mass_matrix: Array) -> float:
    """Compute kinetic energy: 0.5 * p^T M^{-1} p.

    Parameters
    ----------
    momentum : array, shape (D,)
    inverse_mass_matrix : array, shape (D,) or (D,D)

    Returns
    -------
    float
        Kinetic energy.
    """
    if inverse_mass_matrix.ndim == 1:
        return 0.5 * jnp.dot(momentum, inverse_mass_matrix * momentum)
    else:
        return 0.5 * jnp.dot(momentum, inverse_mass_matrix @ momentum)


def generate_momentum(rng_key, inverse_mass_matrix: Array) -> Array:
    """Sample momentum from the kinetic energy distribution: p ~ N(0, M).

    Parameters
    ----------
    rng_key : PRNGKey
    inverse_mass_matrix : array, shape (D,) or (D,D)

    Returns
    -------
    array, shape (D,)
        Sampled momentum.
    """
    if inverse_mass_matrix.ndim == 1:
        # Diagonal: M = diag(1/inv_mass), so p ~ N(0, diag(1/inv_mass))
        # std = sqrt(1/inv_mass)
        std = jnp.sqrt(1.0 / inverse_mass_matrix)
        return jax.random.normal(rng_key, shape=inverse_mass_matrix.shape) * std
    else:
        # Dense: M = inv(inverse_mass_matrix)
        # We need L such that M = L L^T, so p = L @ z, z ~ N(0, I)
        # M^{-1} = inverse_mass_matrix, so M = inv(inverse_mass_matrix)
        # Cholesky of M: but it's cheaper to use Cholesky of M^{-1}
        # If M^{-1} = L_inv L_inv^T, then M = L_inv^{-T} L_inv^{-1}
        # So p = L_inv^{-T} @ z = solve(L_inv^T, z)
        dim = inverse_mass_matrix.shape[0]
        z = jax.random.normal(rng_key, shape=(dim,))
        L_inv = jnp.linalg.cholesky(inverse_mass_matrix)
        return jax.scipy.linalg.solve_triangular(L_inv.T, z, lower=False)


def total_energy(log_prob: float, momentum: Array, inverse_mass_matrix: Array) -> float:
    """Compute the Hamiltonian H = -log_prob + kinetic_energy."""
    return -log_prob + kinetic_energy(momentum, inverse_mass_matrix)
