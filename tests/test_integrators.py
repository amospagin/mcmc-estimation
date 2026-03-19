"""Tests for Hamiltonian integrators."""

import jax
import jax.numpy as jnp
import pytest

from ag.types import IntegratorState
from ag.integrators.leapfrog import leapfrog_step
from ag.integrators.utils import kinetic_energy, generate_momentum, total_energy


def _standard_normal_logdensity(x):
    """log p(x) for x ~ N(0, I)."""
    return -0.5 * jnp.sum(x ** 2)


class TestLeapfrog:
    def test_energy_conservation_diagonal_mass(self):
        """Leapfrog should approximately conserve energy on a quadratic."""
        dim = 5
        inv_mass = jnp.ones(dim)
        x0 = jnp.array([1.0, -0.5, 0.3, -1.2, 0.8])
        p0 = jnp.array([0.5, 1.0, -0.3, 0.7, -0.9])
        lp, grad = jax.value_and_grad(_standard_normal_logdensity)(x0)
        state = IntegratorState(x0, p0, lp, grad)

        e0 = total_energy(lp, p0, inv_mass)

        # 50 leapfrog steps with small step size
        step_size = 0.01
        for _ in range(50):
            state = leapfrog_step(state, step_size, _standard_normal_logdensity, inv_mass)

        e1 = total_energy(state.log_prob, state.momentum, inv_mass)
        assert jnp.abs(e1 - e0) < 1e-4, f"Energy drift {jnp.abs(e1 - e0)}"

    def test_energy_conservation_dense_mass(self):
        """Energy conservation with dense mass matrix."""
        dim = 3
        # SPD mass matrix
        A = jnp.array([[2.0, 0.5, 0.0], [0.5, 1.5, 0.3], [0.0, 0.3, 1.0]])
        inv_mass = A

        x0 = jnp.array([1.0, -0.5, 0.3])
        p0 = jnp.array([0.5, 1.0, -0.3])
        lp, grad = jax.value_and_grad(_standard_normal_logdensity)(x0)
        state = IntegratorState(x0, p0, lp, grad)
        e0 = total_energy(lp, p0, inv_mass)

        step_size = 0.01
        for _ in range(50):
            state = leapfrog_step(state, step_size, _standard_normal_logdensity, inv_mass)

        e1 = total_energy(state.log_prob, state.momentum, inv_mass)
        assert jnp.abs(e1 - e0) < 1e-3

    def test_reversibility(self):
        """Leapfrog should be time-reversible."""
        dim = 4
        inv_mass = jnp.ones(dim)
        x0 = jnp.ones(dim) * 0.5
        p0 = jnp.array([0.3, -0.7, 0.1, 0.9])
        lp, grad = jax.value_and_grad(_standard_normal_logdensity)(x0)
        state = IntegratorState(x0, p0, lp, grad)

        step_size = 0.05
        n_steps = 20

        # Forward
        for _ in range(n_steps):
            state = leapfrog_step(state, step_size, _standard_normal_logdensity, inv_mass)

        # Reverse momentum and go back
        state = IntegratorState(state.position, -state.momentum, state.log_prob, state.log_prob_grad)
        for _ in range(n_steps):
            state = leapfrog_step(state, step_size, _standard_normal_logdensity, inv_mass)

        assert jnp.allclose(state.position, x0, atol=1e-5)
        assert jnp.allclose(-state.momentum, p0, atol=1e-5)


class TestKineticEnergy:
    def test_diagonal(self):
        p = jnp.array([1.0, 2.0, 3.0])
        inv_mass = jnp.array([1.0, 0.5, 2.0])
        ke = kinetic_energy(p, inv_mass)
        expected = 0.5 * (1.0 * 1.0 + 2.0 * 0.5 * 2.0 + 3.0 * 2.0 * 3.0)
        assert jnp.allclose(ke, expected)

    def test_dense(self):
        p = jnp.array([1.0, 2.0])
        inv_mass = jnp.eye(2)
        ke = kinetic_energy(p, inv_mass)
        assert jnp.allclose(ke, 0.5 * (1.0 + 4.0))


class TestMomentumGeneration:
    def test_shape(self):
        key = jax.random.key(0)
        inv_mass = jnp.ones(5)
        p = generate_momentum(key, inv_mass)
        assert p.shape == (5,)

    def test_variance_diagonal(self):
        """Momentum variance should match M = diag(1/inv_mass)."""
        key = jax.random.key(42)
        inv_mass = jnp.array([0.5, 2.0, 1.0])
        keys = jax.random.split(key, 10000)
        samples = jax.vmap(generate_momentum, in_axes=(0, None))(keys, inv_mass)
        var = jnp.var(samples, axis=0)
        expected_var = 1.0 / inv_mass  # M = diag(1/inv_mass)
        assert jnp.allclose(var, expected_var, atol=0.1)
