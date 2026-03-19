"""Model specification API.

Users define models by providing a log-density function.
"""

from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array


class Model:
    """A probabilistic model defined by its log-density function.

    Parameters
    ----------
    log_density_fn : callable
        Function mapping a flat parameter array (D,) to a scalar log density.
        Must be JAX-traceable (no Python control flow on parameter values).
    param_names : list[str]
        Names of the parameters, in the same order as the flat array.
    param_dims : dict[str, int], optional
        Mapping from parameter name to its dimension. If None, all are scalar.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> def log_density(params):
    ...     # Simple normal: params = [mu]
    ...     return -0.5 * jnp.sum(params**2)
    >>> model = Model(log_density, param_names=["mu"])
    """

    def __init__(
        self,
        log_density_fn: Callable[[Array], float],
        param_names: list[str],
        param_dims: dict[str, int] | None = None,
    ):
        self.log_density_fn = log_density_fn
        self.param_names = param_names
        self.param_dims = param_dims or {name: 1 for name in param_names}
        self.ndim = sum(self.param_dims.values())

    def unpack(self, flat_params: Array) -> dict[str, Array]:
        """Convert flat parameter array to a named dictionary."""
        result = {}
        idx = 0
        for name in self.param_names:
            dim = self.param_dims[name]
            if dim == 1:
                result[name] = flat_params[idx]
            else:
                result[name] = flat_params[idx : idx + dim]
            idx += dim
        return result

    def pack(self, params_dict: dict[str, Array]) -> Array:
        """Convert named parameter dictionary to a flat array."""
        parts = []
        for name in self.param_names:
            val = jnp.atleast_1d(jnp.asarray(params_dict[name]))
            parts.append(val.ravel())
        return jnp.concatenate(parts)

    @staticmethod
    def from_dict_logdensity(
        log_density_fn: Callable[[dict], float],
        param_names: list[str],
        param_dims: dict[str, int] | None = None,
    ) -> "Model":
        """Create a Model from a log-density that takes a dict of params.

        This wraps the dict-based function to work with flat arrays internally.
        """
        dims = param_dims or {name: 1 for name in param_names}
        ndim = sum(dims.values())

        def flat_log_density(flat_params):
            params = {}
            idx = 0
            for name in param_names:
                dim = dims[name]
                if dim == 1:
                    params[name] = flat_params[idx]
                else:
                    params[name] = flat_params[idx : idx + dim]
                idx += dim
            return log_density_fn(params)

        return Model(flat_log_density, param_names, dims)
