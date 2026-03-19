"""Main sampler orchestrator.

Wires together: Model -> Transform -> Kernel -> Adaptation -> Diagnostics.

The sampling loop:
  1. Initialize chains (vmapped)
  2. For each iteration:
     a. Compose transform with log-density
     b. Take one kernel step in transformed space
     c. Map proposal back to model space
     d. Update adaptation (step size, mass matrix)
     e. Periodically train the flow (score matching)
     f. Check convergence
  3. Return SampleResult
"""

import jax
import jax.numpy as jnp
from jax import Array
from functools import partial

from ag.types import KernelState, SampleResult, ConvergenceState
from ag.model import Model
from ag.kernels import mclmc, nuts
from ag.transforms import base as identity_transform
from ag.adaptation import step_size as ss_adapt, mass_matrix as mm_adapt
from ag.diagnostics import convergence as conv


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
        Number of adaptation steps (all online, no samples discarded
        unless flow training is active).
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
        Target acceptance rate (NUTS only, MCLMC uses energy error).
    transform_module : module, optional
        Transform module (e.g., affine_coupling). Defaults to identity.
    flow_params : pytree, optional
        Pre-trained flow parameters. If None and transform_module is
        provided, initializes randomly.
    flow_train_interval : int
        Train the flow every N steps.
    max_tree_depth : int
        Maximum NUTS tree depth.

    Returns
    -------
    SampleResult
    """
    if transform_module is None:
        transform_module = identity_transform

    rng_key = jax.random.key(seed)
    dim = model.ndim

    # --- Initialize transform ---
    rng_key, flow_key = jax.random.split(rng_key)
    if flow_params is None:
        flow_params = transform_module.init_params(flow_key, dim)

    # --- Initialize chains ---
    rng_key, init_key = jax.random.split(rng_key)
    init_keys = jax.random.split(init_key, num_chains)
    initial_positions = jax.random.normal(init_key, shape=(num_chains, dim)) * 0.1

    # --- Build log-density in transformed space ---
    logdensity_fn = model.log_density_fn
    transformed_logdensity = transform_module.make_transformed_logdensity(
        flow_params, logdensity_fn
    )

    # --- Initialize kernel states ---
    if kernel == "mclmc":
        init_fn = partial(mclmc.init, logdensity_fn=transformed_logdensity)
        init_states = jax.vmap(init_fn)(initial_positions, init_keys)
    else:
        init_fn = partial(nuts.init, logdensity_fn=transformed_logdensity)
        init_states = jax.vmap(init_fn)(initial_positions)

    # --- Initialize adaptation ---
    ss_state = ss_adapt.init(initial_step_size)
    mm_state = mm_adapt.init(dim, dense=dense_mass)

    # --- Sampling loop ---
    total_steps = warmup_steps + num_samples
    all_samples = jnp.zeros((num_chains, num_samples, dim))
    all_log_probs = jnp.zeros((num_chains, num_samples))
    divergence_count = 0
    total_accept = 0.0

    states = init_states
    step_size = initial_step_size
    L = initial_L
    inv_mass = jnp.ones(dim) if not dense_mass else jnp.eye(dim)

    for step in range(total_steps):
        rng_key, step_key = jax.random.split(rng_key)
        step_keys = jax.random.split(step_key, num_chains)

        # Build kernel with current adaptation parameters
        if kernel == "mclmc":
            kern = mclmc.build_kernel(transformed_logdensity, step_size, L)
        else:
            kern = nuts.build_kernel(
                transformed_logdensity, step_size, inv_mass, max_tree_depth
            )

        # Step all chains
        states, infos = jax.vmap(kern)(step_keys, states)

        # --- Adaptation (during warmup) ---
        if step < warmup_steps:
            mean_accept = jnp.mean(infos.acceptance_rate)

            if kernel == "nuts":
                ss_state = ss_adapt.update(ss_state, mean_accept, target_accept)
                step_size = ss_adapt.get_step_size(ss_state)

                # Update mass matrix with all chain positions
                for c in range(num_chains):
                    mm_state = mm_adapt.update(mm_state, states.position[c])
                inv_mass = mm_adapt.get_inverse_mass_matrix(mm_state)

            elif kernel == "mclmc":
                # MCLMC adapts step size based on energy variance
                energy_var = jnp.var(infos.energy)
                # Target: energy variance ≈ d (for well-tuned ε)
                energy_error = jnp.sqrt(energy_var) / dim
                ss_state = ss_adapt.update(
                    ss_state, 1.0 - energy_error, target_accept=0.9
                )
                step_size = ss_adapt.get_step_size(ss_state)
                # L adapts more slowly — scale with step size
                L = jnp.maximum(L, step_size * jnp.sqrt(dim))

        elif step == warmup_steps and kernel == "nuts":
            # Fix step size at end of warmup
            step_size = ss_adapt.get_step_size(ss_state, final=True)

        # --- Collect samples ---
        sample_idx = step - warmup_steps
        if step >= warmup_steps:
            # Map back to model space if using a transform
            model_positions = jax.vmap(
                partial(transform_module.forward, flow_params)
            )(states.position)
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
    }

    return SampleResult(
        samples=all_samples,
        log_probs=all_log_probs,
        stats=stats,
        convergence_history=conv_state,
        param_names=model.param_names,
    )
