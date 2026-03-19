"""Tests for coordinate transforms."""

import jax
import jax.numpy as jnp
import pytest

from ag.transforms import base as identity_transform
from ag.transforms import affine_coupling


class TestIdentityTransform:
    def test_forward_inverse_roundtrip(self):
        x = jnp.array([1.0, 2.0, 3.0])
        params = identity_transform.init_params(jax.random.key(0), 3)
        z = identity_transform.inverse(params, x)
        x_rec = identity_transform.forward(params, z)
        assert jnp.allclose(x_rec, x)

    def test_log_det_jac_zero(self):
        z = jnp.array([1.0, 2.0])
        params = identity_transform.init_params(jax.random.key(0), 2)
        ldj = identity_transform.log_det_jac(params, z)
        assert ldj == 0.0


class TestAffineCoupling:
    def test_forward_inverse_roundtrip(self):
        """forward(inverse(x)) should recover x."""
        dim = 6
        key = jax.random.key(42)
        params = affine_coupling.init_params(key, dim, n_layers=4)

        x = jnp.array([1.0, -0.5, 2.0, 0.3, -1.0, 0.7])
        z = affine_coupling.inverse(params, x)
        x_rec = affine_coupling.forward(params, z)

        assert jnp.allclose(x_rec, x, atol=1e-5), f"Max error: {jnp.max(jnp.abs(x_rec - x))}"

    def test_inverse_forward_roundtrip(self):
        """inverse(forward(z)) should recover z."""
        dim = 4
        key = jax.random.key(0)
        params = affine_coupling.init_params(key, dim, n_layers=3)

        z = jnp.array([0.5, -1.0, 0.3, 2.0])
        x = affine_coupling.forward(params, z)
        z_rec = affine_coupling.inverse(params, x)

        assert jnp.allclose(z_rec, z, atol=1e-5)

    def test_starts_near_identity(self):
        """With small initialization, the flow should be near-identity."""
        dim = 4
        key = jax.random.key(0)
        params = affine_coupling.init_params(key, dim)

        z = jnp.array([1.0, 2.0, -1.0, 0.5])
        x = affine_coupling.forward(params, z)

        # With small weight init, x ≈ z
        assert jnp.allclose(x, z, atol=0.1)

    def test_log_det_jac_matches_autodiff(self):
        """log|det J| should match JAX's autodiff Jacobian."""
        dim = 4
        key = jax.random.key(1)
        params = affine_coupling.init_params(key, dim, n_layers=3)

        z = jnp.array([0.5, -0.3, 1.0, -0.7])

        # Our efficient computation
        ldj = affine_coupling.log_det_jac(params, z)

        # Autodiff reference
        jac = jax.jacobian(lambda z_: affine_coupling.forward(params, z_))(z)
        ldj_ref = jnp.log(jnp.abs(jnp.linalg.det(jac)))

        assert jnp.allclose(ldj, ldj_ref, atol=1e-4), f"ldj={ldj}, ref={ldj_ref}"

    def test_transformed_logdensity(self):
        """Transformed log-density should include Jacobian correction."""
        dim = 4
        key = jax.random.key(2)
        params = affine_coupling.init_params(key, dim, n_layers=2)

        def logdensity(x):
            return -0.5 * jnp.sum(x ** 2)

        transformed = affine_coupling.make_transformed_logdensity(params, logdensity)

        z = jnp.array([0.3, -0.5, 0.8, -0.2])
        x = affine_coupling.forward(params, z)
        ldj = affine_coupling.log_det_jac(params, z)

        expected = logdensity(x) + ldj
        actual = transformed(z)

        assert jnp.allclose(actual, expected, atol=1e-6)


class TestScoreMatching:
    def test_loss_low_for_identity_on_gaussian(self):
        """Score matching loss should be low when flow is identity
        and target is N(0,I) (score = -x, which matches -z when flow=identity)."""
        from ag.transforms.score_matching import score_matching_loss

        dim = 3

        def logdensity(x):
            return -0.5 * jnp.sum(x ** 2)

        # Identity flow functions
        def fwd(params, z): return z
        def inv(params, x): return x
        def ldj(params, z): return 0.0

        # Samples from N(0,I)
        key = jax.random.key(0)
        positions = jax.random.normal(key, (50, dim))

        loss = score_matching_loss(None, positions, logdensity, fwd, ldj, inv)
        # For identity flow on N(0,I): score = -x, target = -z = -x, loss ≈ 0
        assert loss < 0.1, f"Loss should be near zero, got {loss}"
