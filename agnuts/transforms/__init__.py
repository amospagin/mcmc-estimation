"""Coordinate transforms (geometry layer).

Transforms are diffeomorphisms that reparameterize the posterior to make
it easier to sample.  The key interface:

    forward(params, z) -> x         Map from unconstrained z to model space x
    inverse(params, x) -> z         Map from model space to unconstrained space
    log_det_jac(params, z) -> float Log |det J| of the forward map at z

All transforms must be invertible and differentiable.  The Jacobian
correction ensures the chain always targets the correct distribution
regardless of transform quality.
"""
