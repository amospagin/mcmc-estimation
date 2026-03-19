"""Affine coupling flow for learned reparameterization.

Uses Real-NVP-style affine coupling layers.  Key properties:
  - Triangular Jacobian → O(d) log-det computation
  - Exact inverse (no iterative solve needed)
  - Expressive enough to capture correlations and varying scales

The flow is trained online via score matching during MCMC sampling.
"""

import jax
import jax.numpy as jnp
from jax import Array
from typing import NamedTuple


class AffineCouplingParams(NamedTuple):
    """Parameters for a stack of affine coupling layers."""
    # Each layer has (shift_weights, shift_bias, log_scale_weights, log_scale_bias)
    # Stored as a list of layer param tuples
    layers: list


def init_params(rng_key, dim: int, n_layers: int = 4, hidden_dim: int | None = None):
    """Initialize affine coupling flow parameters.

    Parameters
    ----------
    rng_key : PRNGKey
    dim : int
        Dimensionality of the sampling space.
    n_layers : int
        Number of coupling layers. More layers = more expressive.
    hidden_dim : int, optional
        Hidden dimension of the coupling networks. Defaults to min(2*dim, 64).
    """
    if hidden_dim is None:
        hidden_dim = min(2 * dim, 64)

    layers = []
    for i in range(n_layers):
        rng_key, k1, k2, k3, k4 = jax.random.split(rng_key, 5)
        # Which dimensions are "fixed" vs "transformed" alternates each layer
        d_fixed = dim // 2 if i % 2 == 0 else dim - dim // 2
        d_transform = dim - d_fixed

        # Small MLP: fixed_dims -> (shift, log_scale) for transformed dims
        # Two-layer MLP with tanh activation
        scale = 0.01  # small init so flow starts near identity
        w1 = jax.random.normal(k1, (d_fixed, hidden_dim)) * scale
        b1 = jnp.zeros(hidden_dim)
        w2_shift = jax.random.normal(k2, (hidden_dim, d_transform)) * scale
        b2_shift = jnp.zeros(d_transform)
        w2_scale = jax.random.normal(k3, (hidden_dim, d_transform)) * scale
        b2_scale = jnp.zeros(d_transform)

        layers.append((w1, b1, w2_shift, b2_shift, w2_scale, b2_scale))

    return AffineCouplingParams(layers=layers)


def _coupling_forward(layer_params, z_fixed, z_transform):
    """Apply one coupling layer forward: z_transform -> x_transform."""
    w1, b1, w2_s, b2_s, w2_l, b2_l = layer_params
    h = jnp.tanh(z_fixed @ w1 + b1)
    shift = h @ w2_s + b2_s
    log_scale = h @ w2_l + b2_l
    # Clamp log_scale for numerical stability
    log_scale = jnp.clip(log_scale, -5.0, 5.0)
    x_transform = z_transform * jnp.exp(log_scale) + shift
    return x_transform, log_scale


def _coupling_inverse(layer_params, x_fixed, x_transform):
    """Apply one coupling layer inverse: x_transform -> z_transform."""
    w1, b1, w2_s, b2_s, w2_l, b2_l = layer_params
    h = jnp.tanh(x_fixed @ w1 + b1)
    shift = h @ w2_s + b2_s
    log_scale = h @ w2_l + b2_l
    log_scale = jnp.clip(log_scale, -5.0, 5.0)
    z_transform = (x_transform - shift) * jnp.exp(-log_scale)
    return z_transform, log_scale


def _split(z, layer_idx, dim):
    """Split z into fixed and transform partitions."""
    d_fixed = dim // 2 if layer_idx % 2 == 0 else dim - dim // 2
    return z[:d_fixed], z[d_fixed:]


def _merge(z_fixed, z_transform, layer_idx, dim):
    """Merge fixed and transform partitions back."""
    return jnp.concatenate([z_fixed, z_transform])


def forward(params: AffineCouplingParams, z: Array) -> Array:
    """Forward pass: z (sampling space) -> x (model space)."""
    dim = z.shape[0]
    x = z
    for i, layer_params in enumerate(params.layers):
        x_fixed, x_transform = _split(x, i, dim)
        x_transform, _ = _coupling_forward(layer_params, x_fixed, x_transform)
        x = _merge(x_fixed, x_transform, i, dim)
    return x


def inverse(params: AffineCouplingParams, x: Array) -> Array:
    """Inverse pass: x (model space) -> z (sampling space)."""
    dim = x.shape[0]
    z = x
    for i in reversed(range(len(params.layers))):
        layer_params = params.layers[i]
        z_fixed, z_transform = _split(z, i, dim)
        z_transform, _ = _coupling_inverse(layer_params, z_fixed, z_transform)
        z = _merge(z_fixed, z_transform, i, dim)
    return z


def log_det_jac(params: AffineCouplingParams, z: Array) -> float:
    """Log |det Jacobian| of the forward transform.

    For affine coupling layers, this is just the sum of log_scales
    across all layers (triangular Jacobian).
    """
    dim = z.shape[0]
    ldj = 0.0
    x = z
    for i, layer_params in enumerate(params.layers):
        x_fixed, x_transform = _split(x, i, dim)
        x_transform, log_scale = _coupling_forward(layer_params, x_fixed, x_transform)
        ldj = ldj + jnp.sum(log_scale)
        x = _merge(x_fixed, x_transform, i, dim)
    return ldj


def make_transformed_logdensity(params, logdensity_fn):
    """Create log-density in sampling (z) space with Jacobian correction.

    log π̃(z) = log π(f(z)) + log |det ∂f/∂z|
    """
    def transformed(z):
        x = forward(params, z)
        ldj = log_det_jac(params, z)
        return logdensity_fn(x) + ldj
    return transformed
