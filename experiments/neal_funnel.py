"""Neal's funnel: the acid test for geometry learning.

Neal's funnel is the canonical hard posterior for MCMC. It has the form:
    v ~ N(0, 9)
    x_i ~ N(0, exp(v))   for i = 1, ..., d-1

The difficulty: when v is large and negative, the x dimensions are
tightly constrained (narrow neck of the funnel). When v is large and
positive, they're spread out (wide mouth). No single global step size
or mass matrix works everywhere.

This is exactly the geometry that our self-improving flow should handle:
the flow learns to undo the position-dependent scaling, making the
transformed posterior approximately Gaussian.

We compare:
  1. MCLMC without flow (baseline — should struggle)
  2. MCLMC with affine coupling flow + online score matching (our method)
  3. NUTS without flow (Stan-like baseline)

Success criterion: method 2 should have:
  - Lower R-hat (better mixing)
  - Higher ESS/step (more efficient)
  - Correct marginal statistics (v ~ N(0,9), x_i ~ N(0, exp(v)))
"""

import time
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ag.model import Model
from ag.sampler import sample


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def neal_funnel_logdensity(params):
    """Log-density of Neal's funnel.

    params[0] = v (log-variance parameter)
    params[1:] = x (the funnel dimensions)
    """
    v = params[0]
    x = params[1:]

    # v ~ N(0, 9) => N(0, 3^2)
    log_p_v = -0.5 * v ** 2 / 9.0

    # x_i ~ N(0, exp(v))
    d = x.shape[0]
    log_p_x = -0.5 * d * v - 0.5 * jnp.sum(x ** 2) * jnp.exp(-v)

    return log_p_v + log_p_x


def make_funnel_model(dim: int = 10) -> Model:
    """Create a Neal's funnel model with given dimensionality."""
    param_names = ["v"] + [f"x{i}" for i in range(dim - 1)]
    param_dims = {name: 1 for name in param_names}
    return Model(neal_funnel_logdensity, param_names, param_dims)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_experiment(
    dim: int = 10,
    num_chains: int = 4,
    num_samples: int = 2000,
    warmup_steps: int = 1000,
    seed: int = 42,
):
    """Run all three methods and compare."""

    model = make_funnel_model(dim)
    results = {}

    # --- 1. MCLMC baseline (no flow) ---
    print(f"Running MCLMC baseline (dim={dim})...")
    t0 = time.time()
    res_mclmc = sample(
        model,
        num_chains=num_chains,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        seed=seed,
        kernel="mclmc",
        initial_step_size=0.1,
        initial_L=1.0,
    )
    t_mclmc = time.time() - t0
    results["mclmc_baseline"] = _summarize(res_mclmc, t_mclmc)
    print(f"  Time: {t_mclmc:.1f}s, R-hat max: {results['mclmc_baseline']['rhat_max']:.4f}")

    # --- 2. MCLMC + flow (our method) ---
    print(f"Running MCLMC + flow (dim={dim})...")
    try:
        from ag.transforms import affine_coupling
        t0 = time.time()
        res_flow = sample(
            model,
            num_chains=num_chains,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            seed=seed,
            kernel="mclmc",
            initial_step_size=0.1,
            initial_L=1.0,
            transform_module=affine_coupling,
            flow_train_interval=50,
        )
        t_flow = time.time() - t0
        results["mclmc_flow"] = _summarize(res_flow, t_flow)
        print(f"  Time: {t_flow:.1f}s, R-hat max: {results['mclmc_flow']['rhat_max']:.4f}")
    except Exception as e:
        print(f"  Flow experiment failed: {e}")
        results["mclmc_flow"] = {"error": str(e)}

    # --- 3. NUTS baseline ---
    print(f"Running NUTS baseline (dim={dim})...")
    t0 = time.time()
    res_nuts = sample(
        model,
        num_chains=num_chains,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        seed=seed,
        kernel="nuts",
        initial_step_size=0.1,
    )
    t_nuts = time.time() - t0
    results["nuts_baseline"] = _summarize(res_nuts, t_nuts)
    print(f"  Time: {t_nuts:.1f}s, R-hat max: {results['nuts_baseline']['rhat_max']:.4f}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    _print_comparison(results, dim)

    # --- Save ---
    out_path = Path("results") / f"funnel_dim{dim}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return results


def _summarize(result, wall_time: float) -> dict:
    """Extract key metrics from a SampleResult."""
    samples = np.array(result.samples)
    conv = result.convergence_history

    # True marginals: v ~ N(0, 9), so var(v) = 9
    v_samples = samples[:, :, 0]
    v_mean = float(np.mean(v_samples))
    v_var = float(np.var(v_samples))

    return {
        "wall_time": wall_time,
        "rhat_max": float(jnp.max(conv.rhat)),
        "rhat_mean": float(jnp.mean(conv.rhat)),
        "ess_min": float(jnp.min(conv.bulk_ess)),
        "ess_median": float(jnp.median(conv.bulk_ess)),
        "divergence_count": int(conv.divergence_count),
        "v_mean": v_mean,
        "v_var": v_var,
        "v_mean_error": abs(v_mean - 0.0),
        "v_var_error": abs(v_var - 9.0),
        "is_converged": bool(conv.is_converged),
    }


def _print_comparison(results: dict, dim: int):
    """Print a formatted comparison table."""
    header = f"{'Method':<20} {'Time':>8} {'R-hat':>8} {'ESS min':>8} {'Diverg':>8} {'v_mean':>8} {'v_var':>8}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<20} {'FAILED':>8}")
            continue
        print(
            f"{name:<20} "
            f"{r['wall_time']:>7.1f}s "
            f"{r['rhat_max']:>8.4f} "
            f"{r['ess_min']:>8.0f} "
            f"{r['divergence_count']:>8d} "
            f"{r['v_mean']:>8.3f} "
            f"{r['v_var']:>8.3f}"
        )
    print(f"\nTrue values: v_mean=0.000, v_var=9.000")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Neal's funnel experiment")
    parser.add_argument("--dim", type=int, default=10, help="Funnel dimension")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_experiment(
        dim=args.dim,
        num_chains=args.chains,
        num_samples=args.samples,
        warmup_steps=args.warmup,
        seed=args.seed,
    )
