"""Online score matching for flow training during MCMC.

The core idea: at each MCMC step we have samples x and exact scores
∇ log π(x) from autodiff.  We train the flow to match these scores
so that in the transformed space the target becomes approximately
standard normal.

Score matching loss:
    L(θ) = E_chain[ ||∇_z log q_θ(z) - ∇_z log π̃_θ(z)||² ]

where π̃_θ(z) = π(f_θ(z)) |det Jf_θ(z)| is the pullback density.

If the flow were perfect, ∇_z log π̃_θ(z) = -z (standard normal score).
So we can simplify to:

    L(θ) = E_chain[ ||∇_z log π̃_θ(z) + z||² ]

which pushes the transformed density toward N(0, I).
"""

import jax
import jax.numpy as jnp
from jax import Array
from typing import Any


def score_matching_loss(
    flow_params,
    positions: Array,
    logdensity_fn,
    flow_forward_fn,
    flow_log_det_jac_fn,
    flow_inverse_fn,
):
    """Compute the score matching loss for a batch of positions.

    Parameters
    ----------
    flow_params : pytree
        Current flow parameters.
    positions : Array, shape (batch, D)
        Samples from the chain (in model space x).
    logdensity_fn : callable
        Original log-density function (maps x -> scalar).
    flow_forward_fn : callable
        (params, z) -> x
    flow_log_det_jac_fn : callable
        (params, z) -> log|det J|
    flow_inverse_fn : callable
        (params, x) -> z

    Returns
    -------
    float
        Mean score matching loss over the batch.
    """
    def single_loss(x):
        # Map to sampling space
        z = flow_inverse_fn(flow_params, x)

        # Score of the pullback density in z-space
        def log_pullback(z_):
            x_ = flow_forward_fn(flow_params, z_)
            return logdensity_fn(x_) + flow_log_det_jac_fn(flow_params, z_)

        score_z = jax.grad(log_pullback)(z)

        # Target score for N(0,I) is -z
        # Loss: ||score_z - (-z)||^2 = ||score_z + z||^2
        return jnp.sum((score_z + z) ** 2)

    losses = jax.vmap(single_loss)(positions)
    return jnp.mean(losses)


def train_step(
    flow_params,
    opt_state,
    positions: Array,
    logdensity_fn,
    flow_forward_fn,
    flow_log_det_jac_fn,
    flow_inverse_fn,
    optimizer,
):
    """One gradient step of score matching on accumulated samples.

    Parameters
    ----------
    flow_params : pytree
        Current flow parameters.
    opt_state : pytree
        Optimizer state (e.g., from optax).
    positions : Array, shape (batch, D)
        Batch of positions to train on.
    logdensity_fn, flow_forward_fn, flow_log_det_jac_fn, flow_inverse_fn
        As in score_matching_loss.
    optimizer : optax optimizer
        e.g., optax.adam(1e-3).

    Returns
    -------
    flow_params : updated flow parameters
    opt_state : updated optimizer state
    loss : float, the loss before the update
    """
    loss, grads = jax.value_and_grad(score_matching_loss)(
        flow_params, positions, logdensity_fn,
        flow_forward_fn, flow_log_det_jac_fn, flow_inverse_fn,
    )
    updates, opt_state = optimizer.update(grads, opt_state, flow_params)
    try:
        import optax
        flow_params = optax.apply_updates(flow_params, updates)
    except ImportError:
        # Fallback: manual param update (tree_map)
        flow_params = jax.tree.map(lambda p, u: p + u, flow_params, updates)

    return flow_params, opt_state, loss
