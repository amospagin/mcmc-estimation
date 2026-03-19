"""Online convergence monitoring.

Implements:
  - Split R-hat (Vehtari et al. 2021)
  - Bulk and tail ESS
  - Gradient-informed R-hat (novel): uses gradient statistics across
    chains for faster convergence detection
"""

import jax
import jax.numpy as jnp
from jax import Array

from agsampler.types import ConvergenceState


def init(dim: int) -> ConvergenceState:
    """Initialize convergence state."""
    return ConvergenceState(
        rhat=jnp.full(dim, jnp.inf),
        bulk_ess=jnp.zeros(dim),
        tail_ess=jnp.zeros(dim),
        grad_rhat=jnp.full(dim, jnp.inf),
        divergence_count=0,
        total_steps=0,
        is_converged=False,
    )


def split_rhat(chains: Array) -> Array:
    """Compute split R-hat for each parameter.

    Parameters
    ----------
    chains : Array, shape (num_chains, num_samples, dim)

    Returns
    -------
    Array, shape (dim,)
        R-hat values.  < 1.01 indicates convergence.
    """
    num_chains, num_samples, dim = chains.shape

    # Split each chain in half → 2 * num_chains half-chains
    mid = num_samples // 2
    first_half = chains[:, :mid, :]
    second_half = chains[:, mid:2*mid, :]
    split_chains = jnp.concatenate([first_half, second_half], axis=0)

    m = split_chains.shape[0]  # number of split chains
    n = split_chains.shape[1]  # length of each split chain

    # Between-chain variance B
    chain_means = jnp.mean(split_chains, axis=1)  # (m, dim)
    grand_mean = jnp.mean(chain_means, axis=0)     # (dim,)
    B = n / (m - 1) * jnp.sum((chain_means - grand_mean) ** 2, axis=0)

    # Within-chain variance W
    chain_vars = jnp.var(split_chains, axis=1, ddof=1)  # (m, dim)
    W = jnp.mean(chain_vars, axis=0)                     # (dim,)

    # R-hat
    var_hat = (1.0 - 1.0 / n) * W + B / n
    rhat = jnp.sqrt(var_hat / jnp.maximum(W, 1e-10))

    return rhat


def effective_sample_size(chains: Array) -> Array:
    """Estimate bulk ESS using the autocorrelation method.

    Parameters
    ----------
    chains : Array, shape (num_chains, num_samples, dim)

    Returns
    -------
    Array, shape (dim,)
        Effective sample size per parameter.
    """
    num_chains, num_samples, dim = chains.shape
    total = num_chains * num_samples

    # Simple initial positive sequence estimator
    chain_means = jnp.mean(chains, axis=1, keepdims=True)
    centered = chains - chain_means

    # FFT-based autocorrelation per chain, averaged
    def _chain_autocorr(chain):
        # chain: (num_samples, dim)
        n = chain.shape[0]
        fft_len = 2 * n
        fft_vals = jnp.fft.rfft(chain, n=fft_len, axis=0)
        acf = jnp.fft.irfft(jnp.abs(fft_vals) ** 2, n=fft_len, axis=0)[:n]
        acf = acf / acf[0:1]  # normalize
        return acf

    # Average autocorrelation across chains
    acf = jax.vmap(_chain_autocorr)(centered)  # (num_chains, num_samples, dim)
    mean_acf = jnp.mean(acf, axis=0)           # (num_samples, dim)

    # Sum autocorrelations in pairs until negative
    # Simplified: sum first K positive autocorrelation pairs
    max_lag = num_samples // 2
    tau = jnp.ones(dim)  # integrated autocorrelation time starts at 1
    for lag in range(1, max_lag):
        rho = mean_acf[lag]
        # Only add if positive (conservative cutoff)
        tau = tau + 2.0 * jnp.where(rho > 0, rho, 0.0)

    ess = total / tau
    return ess


def gradient_rhat(
    grad_chains: Array,
) -> Array:
    """Gradient-informed R-hat: apply split R-hat to gradient magnitudes.

    If chains in different regions see systematically different gradient
    statistics, they haven't mixed — even if the sample R-hat looks fine.
    This catches cases where chains are stuck in different modes or
    regions with different local curvature.

    Parameters
    ----------
    grad_chains : Array, shape (num_chains, num_samples, dim)
        Gradients ∇ log π(x) at each sample.

    Returns
    -------
    Array, shape (dim,)
        Gradient R-hat per dimension.
    """
    # Use gradient magnitudes per dimension
    grad_mag = jnp.abs(grad_chains)
    return split_rhat(grad_mag)


def update(
    state: ConvergenceState,
    chains: Array,
    grad_chains: Array | None = None,
    divergent_count: int = 0,
    rhat_threshold: float = 1.01,
    min_ess: float = 400.0,
) -> ConvergenceState:
    """Update convergence state with new chain data.

    Parameters
    ----------
    state : ConvergenceState
    chains : Array, shape (num_chains, num_samples, dim)
        All samples so far.
    grad_chains : Array, optional
        Corresponding gradients (for gradient-informed diagnostics).
    divergent_count : int
        Number of divergent transitions so far.
    rhat_threshold : float
        R-hat convergence threshold.
    min_ess : float
        Minimum ESS for convergence.
    """
    rhat = split_rhat(chains)
    ess = effective_sample_size(chains)

    if grad_chains is not None:
        g_rhat = gradient_rhat(grad_chains)
    else:
        g_rhat = state.grad_rhat

    num_chains, num_samples, _ = chains.shape
    total_steps = num_chains * num_samples

    converged = (
        jnp.all(rhat < rhat_threshold)
        & jnp.all(ess > min_ess)
        & (divergent_count == 0)
    )
    if grad_chains is not None:
        converged = converged & jnp.all(g_rhat < rhat_threshold)

    return ConvergenceState(
        rhat=rhat,
        bulk_ess=ess,
        tail_ess=ess,  # TODO: proper tail ESS
        grad_rhat=g_rhat,
        divergence_count=divergent_count,
        total_steps=total_steps,
        is_converged=converged,
    )
