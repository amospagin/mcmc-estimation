"""Tests for convergence diagnostics."""

import jax
import jax.numpy as jnp
import pytest

from ag.diagnostics import convergence


class TestSplitRhat:
    def test_converged_chains(self):
        """R-hat should be ~1.0 for iid samples from the same distribution."""
        key = jax.random.key(0)
        num_chains, num_samples, dim = 4, 1000, 3
        chains = jax.random.normal(key, (num_chains, num_samples, dim))
        rhat = convergence.split_rhat(chains)
        assert jnp.all(rhat < 1.05), f"R-hat: {rhat}"

    def test_non_converged_chains(self):
        """R-hat should be >> 1 for chains stuck at different values."""
        num_chains, num_samples, dim = 4, 200, 2
        # Each chain has a different mean
        chains = jnp.zeros((num_chains, num_samples, dim))
        for c in range(num_chains):
            chains = chains.at[c, :, :].set(c * 10.0 + 0.01 * jnp.ones((num_samples, dim)))
        rhat = convergence.split_rhat(chains)
        assert jnp.all(rhat > 1.5), f"R-hat should be high: {rhat}"


class TestESS:
    def test_iid_ess_near_total(self):
        """For iid samples, ESS should be close to total sample count."""
        key = jax.random.key(0)
        num_chains, num_samples, dim = 4, 500, 2
        chains = jax.random.normal(key, (num_chains, num_samples, dim))
        ess = convergence.effective_sample_size(chains)
        total = num_chains * num_samples
        # ESS should be at least 50% of total for iid samples
        assert jnp.all(ess > total * 0.5), f"ESS: {ess}, total: {total}"

    def test_correlated_ess_lower(self):
        """Highly autocorrelated chains should have lower ESS."""
        num_chains, num_samples, dim = 4, 500, 1
        key = jax.random.key(1)
        # Random walk: highly autocorrelated
        chains = jnp.zeros((num_chains, num_samples, dim))
        for c in range(num_chains):
            key, subkey = jax.random.split(key)
            steps = jax.random.normal(subkey, (num_samples, dim)) * 0.01
            chains = chains.at[c].set(jnp.cumsum(steps, axis=0))

        ess = convergence.effective_sample_size(chains)
        total = num_chains * num_samples
        # ESS should be much less than total
        assert jnp.all(ess < total * 0.5), f"ESS should be low: {ess}"


class TestGradientRhat:
    def test_converged_gradient_chains(self):
        """Gradient R-hat should be ~1.0 for similar gradient distributions."""
        key = jax.random.key(0)
        num_chains, num_samples, dim = 4, 500, 3
        grad_chains = jax.random.normal(key, (num_chains, num_samples, dim))
        g_rhat = convergence.gradient_rhat(grad_chains)
        assert jnp.all(g_rhat < 1.05), f"Gradient R-hat: {g_rhat}"


class TestConvergenceUpdate:
    def test_convergence_detected(self):
        """Should detect convergence for well-mixed chains."""
        key = jax.random.key(0)
        dim = 2
        chains = jax.random.normal(key, (4, 1000, dim))
        state = convergence.init(dim)
        state = convergence.update(state, chains)
        assert state.is_converged

    def test_non_convergence_detected(self):
        """Should NOT detect convergence for stuck chains."""
        dim = 2
        # Chains with different means
        chains = jnp.zeros((4, 200, dim))
        for c in range(4):
            chains = chains.at[c].set(c * 100.0)
        state = convergence.init(dim)
        state = convergence.update(state, chains)
        assert not state.is_converged
