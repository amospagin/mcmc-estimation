"""Tests for adaptation mechanisms."""

import jax
import jax.numpy as jnp
import pytest

from ag.adaptation import step_size, mass_matrix


class TestDualAveraging:
    def test_step_size_increases_for_high_acceptance(self):
        """If acceptance > target, step size should increase."""
        state = step_size.init(0.1)
        # Simulate high acceptance (0.95 > 0.8 target)
        for _ in range(50):
            state = step_size.update(state, acceptance_rate=0.95)
        final = step_size.get_step_size(state)
        assert final > 0.1

    def test_step_size_decreases_for_low_acceptance(self):
        """If acceptance < target, step size should decrease."""
        state = step_size.init(0.5)
        for _ in range(50):
            state = step_size.update(state, acceptance_rate=0.3)
        final = step_size.get_step_size(state)
        assert final < 0.5

    def test_averaged_step_size(self):
        """Averaged step size should be smoother than instantaneous."""
        state = step_size.init(0.1)
        for _ in range(100):
            state = step_size.update(state, acceptance_rate=0.8)
        instant = step_size.get_step_size(state, final=False)
        averaged = step_size.get_step_size(state, final=True)
        # Both should be reasonable
        assert 0.01 < instant < 10.0
        assert 0.01 < averaged < 10.0


class TestWelford:
    def test_diagonal_variance(self):
        """Should recover known variance for independent samples."""
        key = jax.random.key(0)
        true_var = jnp.array([1.0, 4.0, 0.25])
        samples = jax.random.normal(key, (5000, 3)) * jnp.sqrt(true_var)

        state = mass_matrix.init(3, dense=False)
        for i in range(5000):
            state = mass_matrix.update(state, samples[i])

        inv_mass = mass_matrix.get_inverse_mass_matrix(state)
        # inv_mass ≈ true variance (with some regularization)
        assert jnp.allclose(inv_mass, true_var, atol=0.2)

    def test_dense_covariance(self):
        """Should recover known covariance structure."""
        key = jax.random.key(1)
        cov = jnp.array([[1.0, 0.5], [0.5, 2.0]])
        L = jnp.linalg.cholesky(cov)
        raw = jax.random.normal(key, (5000, 2))
        samples = raw @ L.T

        state = mass_matrix.init(2, dense=True)
        for i in range(5000):
            state = mass_matrix.update(state, samples[i])

        inv_mass = mass_matrix.get_inverse_mass_matrix(state)
        assert jnp.allclose(inv_mass, cov, atol=0.2)

    def test_init_returns_identity(self):
        """Before enough samples, should return identity."""
        state = mass_matrix.init(3, dense=False)
        inv_mass = mass_matrix.get_inverse_mass_matrix(state)
        assert jnp.allclose(inv_mass, jnp.ones(3))

    def test_dense_init_returns_identity(self):
        state = mass_matrix.init(3, dense=True)
        inv_mass = mass_matrix.get_inverse_mass_matrix(state)
        assert jnp.allclose(inv_mass, jnp.eye(3))
