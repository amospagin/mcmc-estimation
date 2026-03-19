"""Leapfrog (Störmer-Verlet) integrator for Hamiltonian dynamics."""

import jax
import jax.numpy as jnp

from ag.types import IntegratorState


def leapfrog_step(
    state: IntegratorState,
    step_size: float,
    logdensity_fn,
    inverse_mass_matrix,
) -> IntegratorState:
    """Perform a single leapfrog integration step.

    Uses the standard three-stage Störmer-Verlet scheme:
        1. Half-step momentum update
        2. Full-step position update
        3. Half-step momentum update

    Parameters
    ----------
    state : IntegratorState
        Current (position, momentum, log_prob, log_prob_grad).
    step_size : float
        Integration step size (epsilon).
    logdensity_fn : callable
        Function mapping position -> log density (unnormalized).
    inverse_mass_matrix : array
        Inverse mass matrix. Shape (D,) for diagonal, (D,D) for dense.

    Returns
    -------
    IntegratorState
        Updated state after one leapfrog step.
    """
    position, momentum, _, log_prob_grad = state

    # Half-step momentum update: p_{1/2} = p_0 + (eps/2) * grad log pi(q_0)
    momentum = momentum + 0.5 * step_size * log_prob_grad

    # Full-step position update: q_1 = q_0 + eps * M^{-1} p_{1/2}
    position = position + step_size * _multiply_inverse_mass(
        inverse_mass_matrix, momentum
    )

    # Evaluate log density and gradient at new position
    log_prob, log_prob_grad = jax.value_and_grad(logdensity_fn)(position)

    # Half-step momentum update: p_1 = p_{1/2} + (eps/2) * grad log pi(q_1)
    momentum = momentum + 0.5 * step_size * log_prob_grad

    return IntegratorState(position, momentum, log_prob, log_prob_grad)


def _multiply_inverse_mass(inverse_mass_matrix, momentum):
    """Compute M^{-1} @ p, handling both diagonal and dense mass matrices."""
    if inverse_mass_matrix.ndim == 1:
        return inverse_mass_matrix * momentum
    else:
        return inverse_mass_matrix @ momentum
