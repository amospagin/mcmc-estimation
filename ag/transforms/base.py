"""Transform protocol and identity transform.

Transforms are pure-function modules (like kernels).  Each exposes:

    init_params(rng_key, dim) -> params
        Initialize transform parameters.

    forward(params, z) -> x
        Map from sampling space to model space.

    inverse(params, x) -> z
        Map from model space to sampling space.

    log_det_jac(params, z) -> float
        Log absolute determinant of the Jacobian of forward at z.

    make_transformed_logdensity(params, logdensity_fn) -> transformed_fn
        Compose the transform with a log-density to get the log-density
        in sampling space (including Jacobian correction).
"""

import jax
import jax.numpy as jnp
from jax import Array


# ---------------------------------------------------------------------------
# Identity transform (no-op baseline)
# ---------------------------------------------------------------------------

def init_params(rng_key, dim: int):
    """Identity transform has no parameters."""
    return None


def forward(params, z: Array) -> Array:
    """Identity: x = z."""
    return z


def inverse(params, x: Array) -> Array:
    """Identity: z = x."""
    return x


def log_det_jac(params, z: Array) -> float:
    """Identity: log|det J| = 0."""
    return 0.0


def make_transformed_logdensity(params, logdensity_fn):
    """With identity transform, the log-density is unchanged."""
    return logdensity_fn
