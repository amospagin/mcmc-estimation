"""Core data structures for the AG sampler.

All types are NamedTuples for JAX pytree compatibility.
Kernel-agnostic: no NUTS-specific or MCLMC-specific types here.
"""

from typing import NamedTuple, Any

import jax.numpy as jnp
from jax import Array


# ---------------------------------------------------------------------------
# Shared state: what every kernel operates on
# ---------------------------------------------------------------------------

class IntegratorState(NamedTuple):
    """State of a Hamiltonian integrator (shared across kernels)."""
    position: Array       # (D,)
    momentum: Array       # (D,)
    log_prob: float
    log_prob_grad: Array  # (D,)


class KernelState(NamedTuple):
    """Minimal state that persists between MCMC transitions.

    This is kernel-agnostic: any kernel must be able to produce and consume
    a KernelState.  Kernel-specific auxiliary data (e.g. MCLMC velocity)
    lives in the `aux` field.
    """
    position: Array       # (D,)
    log_prob: float
    log_prob_grad: Array  # (D,)
    aux: Any = None       # kernel-specific persistent state


class KernelInfo(NamedTuple):
    """Diagnostic information from a single MCMC transition.

    Common fields that all kernels should populate.  Kernel-specific
    diagnostics can be attached via the `extra` field.
    """
    momentum: Array
    is_divergent: bool
    num_integration_steps: int
    acceptance_rate: float
    energy: float
    extra: Any = None         # kernel-specific diagnostics


# ---------------------------------------------------------------------------
# Adaptation state
# ---------------------------------------------------------------------------

class DualAveragingState(NamedTuple):
    """State for dual-averaging step size adaptation (Nesterov 2009)."""
    log_step_size: float
    log_step_size_avg: float
    h_bar: float
    mu: float
    step: int


class WelfordState(NamedTuple):
    """State for Welford online variance/covariance estimation."""
    count: int
    mean: Array        # (D,)
    m2: Array          # (D,) for diagonal, (D,D) for dense


class AdaptationState(NamedTuple):
    """Combined adaptation state."""
    step_size: float
    inverse_mass_matrix: Array  # (D,) diagonal or (D,D) dense
    step: int
    ss_state: DualAveragingState
    mm_state: WelfordState


# ---------------------------------------------------------------------------
# Transform (geometry) state
# ---------------------------------------------------------------------------

class TransformState(NamedTuple):
    """State of a learned coordinate transform (flow).

    `params` holds the flow parameters (pytree), `info` holds training
    diagnostics like the score matching loss history.
    """
    params: Any           # flow parameters (pytree)
    opt_state: Any        # optimizer state
    step: int
    info: Any = None


# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------

class ConvergenceState(NamedTuple):
    """State of the online convergence monitor."""
    rhat: Array               # (D,) per-parameter R-hat
    bulk_ess: Array           # (D,)
    tail_ess: Array           # (D,)
    grad_rhat: Array          # (D,) gradient-informed R-hat
    divergence_count: int
    total_steps: int
    is_converged: bool


# ---------------------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------------------

class SampleResult(NamedTuple):
    """Result of a sampling run."""
    samples: Array             # (num_chains, num_samples, D)
    log_probs: Array           # (num_chains, num_samples)
    stats: dict                # acceptance rates, divergences, etc.
    convergence_history: Any   # R-hat/ESS over time
    param_names: list[str]
