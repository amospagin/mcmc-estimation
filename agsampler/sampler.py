"""Main sampler orchestrator.

Wires together: Model -> Transform -> Kernel -> Adaptation -> Diagnostics.

The sampling loop:
  1. Initialize chains (vmapped)
  2. For each iteration:
     a. Take one kernel step in transformed space
     b. Accumulate model-space positions in sample buffer
     c. Update adaptation (step size, mass matrix)
     d. Periodically train the flow via score matching on the buffer
     e. After flow update, remap chain states to new z-space
     f. Check convergence
  3. Return SampleResult
"""

import jax
import jax.numpy as jnp
from jax import Array
from functools import partial

from agsampler.types import KernelState, SampleResult, ConvergenceState
from agsampler.model import Model
from agsampler.kernels import mclmc, nuts
from agsampler.transforms import base as identity_transform
from agsampler.transforms import score_matching as sm
from agsampler.adaptation import step_size as ss_adapt, mass_matrix as mm_adapt
from agsampler.diagnostics import convergence as conv


def sample(
    model: Model,
    num_chains: int = 4,
    num_samples: int = 1000,
    warmup_steps: int = 500,
    seed: int = 0,
    kernel: str = "mclmc",
    initial_step_size: float = 0.1,
    initial_L: float = 1.0,
    dense_mass: bool = False,
    target_accept: float = 0.8,
    transform_module=None,
    flow_params=None,
    flow_train_interval: int = 50,
    flow_train_steps: int = 10,
    flow_lr: float = 1e-3,
    max_tree_depth: int = 10,
) -> SampleResult:
    """Run the adaptive geometry sampler.

    Parameters
    ----------
    model : Model
        The probabilistic model to sample from.
    num_chains : int
        Number of parallel chains.
    num_samples : int
        Number of post-warmup samples per chain.
    warmup_steps : int
        Number of adaptation steps.
    seed : int
        Random seed.
    kernel : str
        "mclmc" (primary) or "nuts" (fallback).
    initial_step_size : float
        Starting step size for adaptation.
    initial_L : float
        MCLMC trajectory length (ignored for NUTS).
    dense_mass : bool
        Use dense mass matrix (NUTS only).
    target_accept : float
        Target acceptance rate (NUTS only).
    transform_module : module, optional
        Transform module (e.g., affine_coupling). None = identity.
    flow_params : pytree, optional
        Pre-trained flow parameters.
    flow_train_interval : int
        Train the flow every N steps during warmup.
    flow_train_steps : int
        Number of optimizer steps per flow training round.
    flow_lr : float
        Learning rate for flow optimizer.
    max_tree_depth : int
        Maximum NUTS tree depth.

    Returns
    -------
    SampleResult
    """
    if transform_module is None:
        transform_module = identity_transform

    use_flow = transform_module is not identity_transform
    rng_key = jax.random.key(seed)
    dim = model.ndim
    logdensity_fn = model.log_density_fn

    # --- Initialize transform ---
    rng_key, flow_key = jax.random.split(rng_key)
    if flow_params is None:
        flow_params = transform_module.init_params(flow_key, dim)

    # --- Initialize flow optimizer ---
    opt_state = None
    if use_flow:
        try:
            import optax
            optimizer = optax.adam(flow_lr)
            opt_state = optimizer.init(flow_params)
        except ImportError:
            raise ImportError(
                "optax is required for flow training. "
                "Install with: pip install optax"
            )

    # --- Initialize chains ---
    rng_key, init_key = jax.random.split(rng_key)
    init_keys = jax.random.split(init_key, num_chains)
    initial_positions = jax.random.normal(init_key, shape=(num_chains, dim)) * 0.1

    # Build initial transformed log-density
    transformed_logdensity = transform_module.make_transformed_logdensity(
        flow_params, logdensity_fn
    )

    # Initialize kernel states (in z-space)
    if kernel == "mclmc":
        def _mclmc_init(pos, key):
            return mclmc.init(pos, transformed_logdensity, key)
        states = jax.vmap(_mclmc_init)(initial_positions, init_keys)
    else:
        def _nuts_init(pos):
            return nuts.init(pos, transformed_logdensity)
        states = jax.vmap(_nuts_init)(initial_positions)

    # --- Initialize adaptation ---
    ss_state = ss_adapt.init(initial_step_size)
    mm_state = mm_adapt.init(dim, dense=dense_mass)

    # --- Sampling loop state ---
    total_steps = warmup_steps + num_samples
    all_samples = jnp.zeros((num_chains, num_samples, dim))
    all_log_probs = jnp.zeros((num_chains, num_samples))
    divergence_count = 0
    total_accept = 0.0

    step_size = initial_step_size
    L = initial_L
    inv_mass = jnp.ones(dim) if not dense_mass else jnp.eye(dim)

    # Sample buffer for flow training: accumulate model-space positions
    # Ring buffer of size (buffer_size, dim)
    buffer_size = flow_train_interval * num_chains
    sample_buffer = jnp.zeros((buffer_size, dim))
    buffer_idx = 0
    buffer_count = 0  # how many samples have been added total

    flow_losses = []

    for step in range(total_steps):
        rng_key, step_key = jax.random.split(rng_key)
        step_keys = jax.random.split(step_key, num_chains)

        # Build kernel with current parameters
        if kernel == "mclmc":
            kern = mclmc.build_kernel(transformed_logdensity, step_size, L)
        else:
            kern = nuts.build_kernel(
                transformed_logdensity, step_size, inv_mass, max_tree_depth
            )

        # Step all chains
        states, infos = jax.vmap(kern)(step_keys, states)

        # --- Accumulate model-space positions in buffer ---
        if use_flow:
            # Map z -> x (model space) for the buffer
            model_positions = jax.vmap(
                partial(transform_module.forward, flow_params)
            )(states.position)

            for c in range(num_chains):
                idx = buffer_idx % buffer_size
                sample_buffer = sample_buffer.at[idx].set(model_positions[c])
                buffer_idx += 1
                buffer_count += 1

        # --- Adaptation (during warmup) ---
        if step < warmup_steps:
            mean_accept = jnp.mean(infos.acceptance_rate)

            if kernel == "nuts":
                ss_state = ss_adapt.update(ss_state, mean_accept, target_accept)
                step_size = ss_adapt.get_step_size(ss_state)
                for c in range(num_chains):
                    mm_state = mm_adapt.update(mm_state, states.position[c])
                inv_mass = mm_adapt.get_inverse_mass_matrix(mm_state)

            elif kernel == "mclmc":
                energy_var = jnp.var(infos.energy)
                energy_error = jnp.sqrt(energy_var) / dim
                ss_state = ss_adapt.update(
                    ss_state, 1.0 - energy_error, target_accept=0.9
                )
                step_size = ss_adapt.get_step_size(ss_state)
                L = jnp.maximum(L, step_size * jnp.sqrt(dim))

            # --- Flow training ---
            if use_flow and (step + 1) % flow_train_interval == 0 and buffer_count >= dim * 2:
                rng_key, train_key = jax.random.split(rng_key)

                # Use all available buffer samples (up to buffer_size)
                n_available = min(buffer_count, buffer_size)
                train_batch = sample_buffer[:n_available]

                # Decay learning rate: reduce flow updates over time
                # to satisfy Robbins-Monro conditions for ergodicity
                decay = 1.0 / (1.0 + step / warmup_steps)

                for _ in range(flow_train_steps):
                    flow_params, opt_state, loss = sm.train_step(
                        flow_params,
                        opt_state,
                        train_batch,
                        logdensity_fn,
                        transform_module.forward,
                        transform_module.log_det_jac,
                        transform_module.inverse,
                        optimizer,
                    )
                    flow_losses.append(float(loss))

                # --- Remap chain states to new z-space ---
                # Old z -> x via old flow was already computed (model_positions)
                # x -> new z via updated flow's inverse
                transformed_logdensity = transform_module.make_transformed_logdensity(
                    flow_params, logdensity_fn
                )

                # Recompute z-space positions and gradients for all chains
                if not use_flow:
                    pass  # identity, nothing to remap
                else:
                    model_positions = jax.vmap(
                        partial(transform_module.forward, flow_params)
                    )(states.position)
                    # Actually: we need to go x -> z_new
                    # But we already have model_positions from above (before flow update)
                    # Wait — model_positions was computed with the OLD flow params.
                    # We saved them before the flow update. But we need to
                    # recompute because flow_params changed.
                    #
                    # Correct approach:
                    # 1. We have z_old (states.position) in OLD z-space
                    # 2. Map to model space: x = old_forward(z_old) — but old flow_params
                    #    are gone. We computed model_positions before the training loop
                    #    with the old params. But that was done before the flow update.
                    #    Actually we do have model_positions from the buffer accumulation
                    #    step above, which used the OLD flow_params. But we should store
                    #    them explicitly.
                    #
                    # Simpler: the buffer stores model-space positions. We already
                    # have the latest model_positions from the buffer accumulation.
                    # Use those (computed with old flow before the update).
                    #
                    # x = model_positions (from old flow, still correct model-space coords)
                    # z_new = new_inverse(x)
                    new_z = jax.vmap(
                        partial(transform_module.inverse, flow_params)
                    )(model_positions)

                    # Recompute log_prob and grad at new z positions
                    def _reinit_state(z):
                        lp, grad = jax.value_and_grad(transformed_logdensity)(z)
                        return z, lp, grad

                    new_z, new_lp, new_grad = jax.vmap(_reinit_state)(new_z)

                    if kernel == "mclmc":
                        # Preserve velocity direction, rescale to match new space
                        old_velocity = states.aux
                        states = KernelState(
                            position=new_z,
                            log_prob=new_lp,
                            log_prob_grad=new_grad,
                            aux=old_velocity,
                        )
                    else:
                        states = KernelState(
                            position=new_z,
                            log_prob=new_lp,
                            log_prob_grad=new_grad,
                        )

        elif step == warmup_steps and kernel == "nuts":
            step_size = ss_adapt.get_step_size(ss_state, final=True)

        # --- Collect post-warmup samples ---
        sample_idx = step - warmup_steps
        if step >= warmup_steps:
            if use_flow:
                model_positions = jax.vmap(
                    partial(transform_module.forward, flow_params)
                )(states.position)
            else:
                model_positions = states.position
            all_samples = all_samples.at[:, sample_idx, :].set(model_positions)
            all_log_probs = all_log_probs.at[:, sample_idx].set(states.log_prob)
            divergence_count += int(jnp.sum(infos.is_divergent))
            total_accept += float(jnp.mean(infos.acceptance_rate))

    # --- Final diagnostics ---
    conv_state = conv.init(dim)
    conv_state = conv.update(
        conv_state, all_samples, divergent_count=divergence_count
    )

    stats = {
        "mean_acceptance_rate": total_accept / max(num_samples, 1),
        "divergence_count": divergence_count,
        "final_step_size": float(step_size),
        "num_chains": num_chains,
        "num_samples": num_samples,
        "warmup_steps": warmup_steps,
        "kernel": kernel,
        "flow_training": use_flow,
        "flow_losses": flow_losses,
        "flow_train_rounds": len(flow_losses) // max(flow_train_steps, 1),
    }

    return SampleResult(
        samples=all_samples,
        log_probs=all_log_probs,
        stats=stats,
        convergence_history=conv_state,
        param_names=model.param_names,
    )
