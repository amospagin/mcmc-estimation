"""Online score matching for flow training during MCMC.

The core idea: at each MCMC step we have samples x and exact scores
∇ log π(x) from autodiff.  We train the flow to match these scores
so that in the transformed space the target becomes approximately
standard normal.

Score matching loss:
    L(θ) = E_chain[ ||∇_z log π̃_θ(z) + z||² ]

If the flow were perfect, the pullback density would be N(0, I),
whose score is -z.  So this loss pushes the transformed density
toward standard normal.
"""

import jax
import jax.numpy as jnp
from jax import Array


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
        # Loss: ||score_z + z||^2
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
        Optimizer state (from optax).
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
    import optax

    loss, grads = jax.value_and_grad(score_matching_loss)(
        flow_params, positions, logdensity_fn,
        flow_forward_fn, flow_log_det_jac_fn, flow_inverse_fn,
    )
    updates, opt_state = optimizer.update(grads, opt_state, flow_params)
    flow_params = optax.apply_updates(flow_params, updates)

    return flow_params, opt_state, loss
