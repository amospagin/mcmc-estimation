"""Main sampler orchestrator.

Three-phase hybrid warmup strategy:

  Phase 1 — NUTS exploration (when using a flow):
    Run NUTS to get diverse samples and adapt step size + mass matrix.
    NUTS adapts trajectory length per-step, so it can explore even with
    bad geometry.  Samples are accumulated in a buffer.

  Phase 2 — Flow training:
    Train the affine coupling flow via score matching on the NUTS samples.
    This teaches the flow the posterior geometry.  May repeat: alternate
    between short NUTS runs and flow training rounds.

  Phase 3 — Production sampling:
    Run the selected kernel (MCLMC or NUTS) in the flow-transformed space.
    The posterior in transformed space is approximately Gaussian, so
    MCLMC is near-optimal.  Continue refining the flow periodically.

Without a flow, the sampler runs the selected kernel with standard
step size and mass matrix adaptation (equivalent to a typical NUTS/MCLMC
sampler).
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
    nuts_warmup_steps: int | None = None,
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
        Total warmup steps (includes NUTS warmup when using a flow).
    seed : int
        Random seed.
    kernel : str
        "mclmc" (primary) or "nuts" (fallback) for production sampling.
    initial_step_size : float
        Starting step size.
    initial_L : float
        MCLMC trajectory length.
    dense_mass : bool
        Use dense mass matrix.
    target_accept : float
        Target acceptance rate for NUTS.
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
    nuts_warmup_steps : int, optional
        Number of NUTS warmup steps before switching to the production
        kernel.  Defaults to warmup_steps // 2 when using a flow.
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

    # Default NUTS warmup: half of total warmup when using flow
    if nuts_warmup_steps is None:
        nuts_warmup_steps = warmup_steps // 2 if use_flow else 0

    # --- Initialize transform ---
    rng_key, flow_key = jax.random.split(rng_key)
    if flow_params is None:
        flow_params = transform_module.init_params(flow_key, dim)

    # --- Initialize flow optimizer ---
    opt_state = None
    optimizer = None
    if use_flow:
        import optax
        optimizer = optax.adam(flow_lr)
        opt_state = optimizer.init(flow_params)

    # --- Initialize chains ---
    rng_key, init_key = jax.random.split(rng_key)
    init_keys = jax.random.split(init_key, num_chains)
    # Wider initialization to help explore (important for funnels)
    initial_positions = jax.random.normal(init_key, shape=(num_chains, dim))

    # --- Phase 1: NUTS warmup for exploration ---
    # (skipped when not using a flow, or when kernel is already NUTS
    #  and no separate NUTS warmup is requested)

    inv_mass = jnp.ones(dim) if not dense_mass else jnp.eye(dim)
    step_size = initial_step_size
    ss_state = ss_adapt.init(initial_step_size)
    mm_state = mm_adapt.init(dim, dense=dense_mass)

    # Sample buffer: stores model-space positions for flow training
    max_buffer = max(warmup_steps * num_chains, 1)
    sample_buffer = []

    flow_losses = []

    if nuts_warmup_steps > 0 and use_flow:
        print(f"  Phase 1: NUTS warmup ({nuts_warmup_steps} steps)...")

        # Initialize NUTS states (no transform yet — work in model space)
        def _nuts_init(pos):
            return nuts.init(pos, logdensity_fn)
        nuts_states = jax.vmap(_nuts_init)(initial_positions)

        for step in range(nuts_warmup_steps):
            rng_key, step_key = jax.random.split(rng_key)
            step_keys = jax.random.split(step_key, num_chains)

            kern = nuts.build_kernel(
                logdensity_fn, step_size, inv_mass, max_tree_depth
            )
            nuts_states, infos = jax.vmap(kern)(step_keys, nuts_states)

            # Adapt step size and mass matrix
            mean_accept = jnp.mean(infos.acceptance_rate)
            ss_state = ss_adapt.update(ss_state, mean_accept, target_accept)
            step_size = ss_adapt.get_step_size(ss_state)
            for c in range(num_chains):
                mm_state = mm_adapt.update(mm_state, nuts_states.position[c])
            inv_mass = mm_adapt.get_inverse_mass_matrix(mm_state)

            # Accumulate model-space positions
            for c in range(num_chains):
                sample_buffer.append(nuts_states.position[c])

            # Periodic flow training during NUTS warmup
            if (step + 1) % flow_train_interval == 0 and len(sample_buffer) >= dim * 2:
                flow_params, opt_state, new_losses = _train_flow(
                    flow_params, opt_state, sample_buffer,
                    logdensity_fn, transform_module, optimizer,
                    flow_train_steps,
                )
                flow_losses.extend(new_losses)

        # Final flow training on all NUTS samples
        if len(sample_buffer) >= dim * 2:
            print(f"  Phase 2: Training flow on {len(sample_buffer)} NUTS samples...")
            flow_params, opt_state, new_losses = _train_flow(
                flow_params, opt_state, sample_buffer,
                logdensity_fn, transform_module, optimizer,
                flow_train_steps * 5,  # more steps for the final round
            )
            flow_losses.extend(new_losses)
            if new_losses:
                print(f"    Flow loss: {new_losses[0]:.2f} -> {new_losses[-1]:.2f}")

        # Extract model-space positions from NUTS states for handoff
        nuts_model_positions = nuts_states.position

        # Remaining warmup steps for the production kernel
        remaining_warmup = warmup_steps - nuts_warmup_steps

        # Fix NUTS step size
        step_size = ss_adapt.get_step_size(ss_state, final=True)
    else:
        nuts_model_positions = initial_positions
        remaining_warmup = warmup_steps

    # --- Phase 3: Production sampling ---
    # Build transformed log-density with trained flow
    transformed_logdensity = transform_module.make_transformed_logdensity(
        flow_params, logdensity_fn
    )

    # Initialize production kernel states in z-space
    if use_flow:
        # Map NUTS positions (model space) -> z-space via flow inverse
        z_positions = jax.vmap(
            partial(transform_module.inverse, flow_params)
        )(nuts_model_positions)
    else:
        z_positions = nuts_model_positions

    if kernel == "mclmc":
        rng_key, vel_key = jax.random.split(rng_key)
        vel_keys = jax.random.split(vel_key, num_chains)
        def _mclmc_init(pos, key):
            return mclmc.init(pos, transformed_logdensity, key)
        states = jax.vmap(_mclmc_init)(z_positions, vel_keys)

        # Reset step size adaptation for MCLMC
        step_size = initial_step_size
        ss_state = ss_adapt.init(initial_step_size)
        L = initial_L
    else:
        def _nuts_init_z(pos):
            return nuts.init(pos, transformed_logdensity)
        states = jax.vmap(_nuts_init_z)(z_positions)
        L = initial_L

    if kernel == "mclmc":
        print(f"  Phase 3: MCLMC sampling ({remaining_warmup} warmup + {num_samples} samples)...")
    else:
        print(f"  Phase 3: NUTS sampling ({remaining_warmup} warmup + {num_samples} samples)...")

    total_steps = remaining_warmup + num_samples
    all_samples = jnp.zeros((num_chains, num_samples, dim))
    all_log_probs = jnp.zeros((num_chains, num_samples))
    divergence_count = 0
    total_accept = 0.0

    # Clear buffer for production-phase flow refinement
    sample_buffer = []

    for step in range(total_steps):
        rng_key, step_key = jax.random.split(rng_key)
        step_keys = jax.random.split(step_key, num_chains)

        # Build kernel
        if kernel == "mclmc":
            kern = mclmc.build_kernel(transformed_logdensity, step_size, L)
        else:
            kern = nuts.build_kernel(
                transformed_logdensity, step_size, inv_mass, max_tree_depth
            )

        # Step all chains
        states, infos = jax.vmap(kern)(step_keys, states)

        # Get model-space positions
        if use_flow:
            model_positions = jax.vmap(
                partial(transform_module.forward, flow_params)
            )(states.position)
        else:
            model_positions = states.position

        # Adaptation during remaining warmup
        if step < remaining_warmup:
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

            # Accumulate for flow refinement
            if use_flow:
                for c in range(num_chains):
                    sample_buffer.append(model_positions[c])

            # Periodic flow refinement
            if use_flow and (step + 1) % flow_train_interval == 0 and len(sample_buffer) >= dim * 2:
                # Save old model positions before flow update
                old_model_positions = model_positions

                flow_params, opt_state, new_losses = _train_flow(
                    flow_params, opt_state, sample_buffer,
                    logdensity_fn, transform_module, optimizer,
                    flow_train_steps,
                )
                flow_losses.extend(new_losses)

                # Rebuild transformed log-density
                transformed_logdensity = transform_module.make_transformed_logdensity(
                    flow_params, logdensity_fn
                )

                # Remap chain states: x (old model positions) -> new z
                new_z = jax.vmap(
                    partial(transform_module.inverse, flow_params)
                )(old_model_positions)

                def _reinit_state(z):
                    lp, grad = jax.value_and_grad(transformed_logdensity)(z)
                    return z, lp, grad

                new_z, new_lp, new_grad = jax.vmap(_reinit_state)(new_z)

                if kernel == "mclmc":
                    states = KernelState(
                        position=new_z,
                        log_prob=new_lp,
                        log_prob_grad=new_grad,
                        aux=states.aux,  # preserve velocity
                    )
                else:
                    states = KernelState(
                        position=new_z,
                        log_prob=new_lp,
                        log_prob_grad=new_grad,
                    )

        elif step == remaining_warmup:
            if kernel == "nuts":
                step_size = ss_adapt.get_step_size(ss_state, final=True)

        # Collect post-warmup samples
        sample_idx = step - remaining_warmup
        if step >= remaining_warmup:
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


def _train_flow(
    flow_params, opt_state, sample_buffer,
    logdensity_fn, transform_module, optimizer,
    n_steps,
):
    """Run score matching training on buffered samples.

    Returns updated flow_params, opt_state, and list of losses.
    """
    train_batch = jnp.stack(sample_buffer)
    losses = []

    for _ in range(n_steps):
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
        losses.append(float(loss))

    return flow_params, opt_state, losses
