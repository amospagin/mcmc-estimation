"""Dual averaging step size adaptation (Nesterov 2009, Hoffman & Gelman 2014).

Adjusts step size to target a desired acceptance rate (typically 0.8 for
NUTS, 0.65 for HMC, or tuned for MCLMC energy error).
"""

import jax.numpy as jnp

from ag.types import DualAveragingState


def init(initial_step_size: float) -> DualAveragingState:
    """Initialize dual averaging state."""
    return DualAveragingState(
        log_step_size=jnp.log(initial_step_size),
        log_step_size_avg=0.0,
        h_bar=0.0,
        mu=jnp.log(10.0 * initial_step_size),
        step=1,
    )


def update(
    state: DualAveragingState,
    acceptance_rate: float,
    target_accept: float = 0.8,
    gamma: float = 0.05,
    t0: float = 10.0,
    kappa: float = 0.75,
) -> DualAveragingState:
    """Update step size given observed acceptance rate.

    Parameters
    ----------
    state : DualAveragingState
    acceptance_rate : float
        Observed acceptance probability (or energy error proxy for MCLMC).
    target_accept : float
        Target acceptance rate.
    gamma, t0, kappa : float
        Dual averaging hyperparameters (Hoffman & Gelman defaults).

    Returns
    -------
    DualAveragingState
        Updated state with new step size.
    """
    t = state.step
    eta = 1.0 / (t + t0)
    h_bar = (1.0 - eta) * state.h_bar + eta * (target_accept - acceptance_rate)

    log_step_size = state.mu - (jnp.sqrt(t) / gamma) * h_bar
    t_kappa = t ** (-kappa)
    log_step_size_avg = t_kappa * log_step_size + (1.0 - t_kappa) * state.log_step_size_avg

    return DualAveragingState(
        log_step_size=log_step_size,
        log_step_size_avg=log_step_size_avg,
        h_bar=h_bar,
        mu=state.mu,
        step=t + 1,
    )


def get_step_size(state: DualAveragingState, final: bool = False) -> float:
    """Extract current step size.

    Parameters
    ----------
    final : bool
        If True, return the averaged step size (for production sampling).
        If False, return the current step size (for adaptation).
    """
    if final:
        return jnp.exp(state.log_step_size_avg)
    return jnp.exp(state.log_step_size)
