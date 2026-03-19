"""Microcanonical Langevin Monte Carlo (MCLMC) kernel.

Based on Robnik & Seljak (2023).  Key properties:
  - No accept/reject step — stays on constant-energy surface
  - Partial velocity refresh randomizes direction, preserves speed
  - O(d^{1/4}) scaling on well-conditioned targets
  - No mass matrix needed (preconditioning happens in the transform layer)

This is the primary dynamics engine.  On well-conditioned targets
(i.e., after the flow transform), MCLMC is near-optimal.
"""

import jax
import jax.numpy as jnp
from jax import Array

from agsampler.types import KernelState, KernelInfo, IntegratorState


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init(position: Array, logdensity_fn, rng_key) -> KernelState:
    """Initialize MCLMC state with a random velocity.

    The velocity magnitude is set to sqrt(d) so that the expected
    kinetic energy is d/2 (matching the typical set for a d-dimensional
    standard normal).
    """
    log_prob, log_prob_grad = jax.value_and_grad(logdensity_fn)(position)
    dim = position.shape[0]

    # Random velocity with |v| = sqrt(d)
    velocity = jax.random.normal(rng_key, shape=(dim,))
    velocity = velocity * jnp.sqrt(dim) / jnp.sqrt(jnp.sum(velocity ** 2))

    return KernelState(
        position=position,
        log_prob=log_prob,
        log_prob_grad=log_prob_grad,
        aux=velocity,  # velocity persists between steps
    )


# ---------------------------------------------------------------------------
# Kernel builder
# ---------------------------------------------------------------------------

def build_kernel(logdensity_fn, step_size: float, L: float):
    """Build the MCLMC step function.

    Parameters
    ----------
    logdensity_fn : callable
        Maps position (D,) -> scalar log-density.
    step_size : float
        Integration step size (epsilon).
    L : float
        Effective trajectory length.  Controls the velocity randomization
        angle: delta = step_size * sqrt(d-1) / L.

    Returns
    -------
    step_fn : callable
        (rng_key, KernelState) -> (KernelState, KernelInfo)
    """

    def step_fn(rng_key, state: KernelState) -> tuple[KernelState, KernelInfo]:
        position, log_prob, grad, velocity = state
        dim = position.shape[0]

        # --- Leapfrog integration (no mass matrix) ---
        velocity = velocity + 0.5 * step_size * grad
        position = position + step_size * velocity
        log_prob, grad = jax.value_and_grad(logdensity_fn)(position)
        velocity = velocity + 0.5 * step_size * grad

        # --- Partial velocity refresh ---
        # Rotate velocity by angle delta in a random plane containing v
        delta = step_size * jnp.sqrt(jnp.maximum(dim - 1, 1.0)) / L
        velocity = _partially_refresh_velocity(rng_key, velocity, delta)

        energy = -log_prob + 0.5 * jnp.sum(velocity ** 2)

        new_state = KernelState(
            position=position,
            log_prob=log_prob,
            log_prob_grad=grad,
            aux=velocity,
        )

        info = KernelInfo(
            momentum=velocity,
            is_divergent=False,  # MCLMC has no divergences
            num_integration_steps=1,
            acceptance_rate=1.0,  # no accept/reject
            energy=energy,
        )

        return new_state, info

    return step_fn


# ---------------------------------------------------------------------------
# Velocity refresh
# ---------------------------------------------------------------------------

def _partially_refresh_velocity(
    rng_key, velocity: Array, delta: float
) -> Array:
    """Partially refresh velocity direction while preserving speed.

    Rotates v by angle delta in a random plane containing v:
        v_new = v * cos(delta) + |v| * nu * sin(delta)
    where nu is a random unit vector orthogonal to v.
    """
    speed = jnp.sqrt(jnp.sum(velocity ** 2))
    v_hat = velocity / jnp.maximum(speed, 1e-10)

    # Random vector, then project out the v component
    xi = jax.random.normal(rng_key, shape=velocity.shape)
    xi = xi - jnp.dot(xi, v_hat) * v_hat
    xi_norm = jnp.sqrt(jnp.sum(xi ** 2))
    nu = xi / jnp.maximum(xi_norm, 1e-10)  # unit vector orthogonal to v

    v_new = velocity * jnp.cos(delta) + speed * nu * jnp.sin(delta)
    return v_new
