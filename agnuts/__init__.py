"""AG: Adaptive Geometry sampler for Bayesian inference.

Combines learned reparameterization (normalizing flows) with fast dynamics
(MCLMC / NUTS) and online adaptation to create a self-improving MCMC sampler.
"""

from agnuts.model import Model

__version__ = "0.1.0"
__all__ = ["Model"]
