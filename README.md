# AG — Adaptive Geometry Sampler

A self-improving MCMC sampler that learns the posterior geometry during sampling. Built in JAX.

AG combines **learned reparameterization** (normalizing flows trained via online score matching) with **fast dynamics** (MCLMC / NUTS) to create a feedback loop: better geometry leads to faster mixing leads to better training data leads to better geometry.

## Why

Every MCMC sampler fights posterior geometry — funnels, correlations, varying curvature. Current approaches:

| Sampler | Geometry handling | Limitation |
|---|---|---|
| **Stan** | Global mass matrix, fixed after warmup | Can't adapt to position-dependent curvature |
| **BlackJAX** | Building blocks, no integrated system | Assembly required |
| **NumPyro** | Stan-style adaptation in JAX | Same global-metric limitation |
| **NeuTra-HMC** | Flow-based, but two-phase (VI then HMC) | Flow never improves from MCMC samples |

AG closes the loop: the flow trains *during* MCMC using the exact score (grad log pi) available at every sample. No separate variational phase. The chain always targets the correct distribution regardless of flow quality (Jacobian correction ensures exactness).

## Architecture

```
                Orchestrator
          (convergence, stopping)
                    |
    +---------Sampling Loop----------+
    |                                |
    |   Geometry (Transform)         |
    |     identity / affine flow     |
    |     trained online via         |
    |     score matching             |
    |               |                |
    |   Dynamics (Kernel)            |
    |     MCLMC (primary)            |
    |     NUTS  (fallback)           |
    |               |                |
    |   Adaptation                   |
    |     step size (dual avg)       |
    |     mass matrix (Welford)      |
    |     flow params (score match)  |
    |                                |
    +--------------------------------+
                    |
                  Model
          (log-density, JAX AD)
```

**Kernels are interchangeable.** The transform composes with the log-density before the kernel sees it — kernels don't know about transforms. This keeps each layer simple and testable.

## Key ideas

**1. Online score matching.** At each step we have a sample `x` and the exact score `grad log pi(x)` from autodiff. We train the flow to make the transformed posterior approximately N(0, I) by minimizing `||grad_z log pi_tilde(z) + z||^2`. This works even before the chain has mixed — early samples still provide useful geometry information about explored regions.

**2. MCLMC as primary dynamics.** Microcanonical Langevin (Robnik & Seljak 2023) avoids accept/reject entirely by staying on a constant-energy surface. On a well-conditioned target (which the flow provides), it achieves O(d^{1/4}) scaling — near-optimal.

**3. Gradient-informed convergence.** Standard R-hat looks only at samples. We also compute R-hat on gradient magnitudes across chains. If chains see systematically different gradient statistics, they haven't mixed — even if sample R-hat looks fine. This catches mode-separation failures earlier.

## Quick start

```python
import jax.numpy as jnp
from agsampler import Model
from agsampler.sampler import sample

# Define a model via its log-density
def log_density(params):
    return -0.5 * jnp.sum(params ** 2)

model = Model(log_density, param_names=["x", "y"], param_dims={"x": 1, "y": 1})

# Sample with MCLMC (default)
result = sample(model, num_chains=4, num_samples=1000)

print(result.samples.shape)           # (4, 1000, 2)
print(result.convergence_history.rhat) # should be near 1.0
```

With a learned flow:

```python
from agsampler.transforms import affine_coupling
from agsampler.sampler import sample

result = sample(
    model,
    kernel="mclmc",
    transform_module=affine_coupling,
    flow_train_interval=50,
    num_samples=2000,
)
```

## Project structure

```
agsampler/
  model.py                      Model API (log-density wrapper)
  sampler.py                    Main entry point — orchestrates everything
  types.py                      Kernel-agnostic state types
  kernels/
    mclmc.py                    Microcanonical Langevin (primary)
    nuts.py                     No-U-Turn Sampler (fallback)
    base.py                     Kernel protocol docs
  transforms/
    affine_coupling.py          Real-NVP flow (O(d) Jacobian)
    score_matching.py           Online score matching loss + training
    base.py                     Identity transform (no-op baseline)
  adaptation/
    step_size.py                Dual averaging
    mass_matrix.py              Welford online covariance
  diagnostics/
    convergence.py              Split R-hat, ESS, gradient-informed R-hat
tests/
  test_integrators.py           Energy conservation, reversibility
  test_kernels.py               MCLMC + NUTS correctness on N(0,I)
  test_transforms.py            Flow roundtrips, Jacobian, score matching
  test_adaptation.py            Dual averaging, Welford variance recovery
  test_diagnostics.py           R-hat, ESS, convergence detection
```

## Install

```bash
pip install -e ".[dev]"       # CPU
pip install -e ".[dev,gpu]"   # GPU (CUDA 12)
pip install -e ".[dev,flow]"  # with optax/equinox for flow training
```

## Run tests

```bash
pytest tests/
```

## Status

Early-stage research prototype. The core components (kernels, transforms, adaptation, diagnostics) are implemented. The self-improving feedback loop (score-matched flow training during MCLMC) is the primary research contribution and is under active development.

### Implemented
- MCLMC kernel with partial velocity refresh
- NUTS kernel with multinomial tree-doubling
- Affine coupling flow (Real-NVP) with efficient Jacobian
- Online score matching loss and training step
- Dual averaging step size adaptation
- Welford online mass matrix estimation
- Split R-hat, ESS, gradient-informed R-hat
- Sampler orchestrator with transform composition

### Next
- Validate the feedback loop on Neal's funnel
- Position-dependent preconditioning (amortized Riemannian)
- Cross-chain adaptation (ChEES-style)
- Benchmarks against Stan, BlackJAX, NumPyro

## References

- Robnik & Seljak (2023). *Microcanonical Langevin Monte Carlo.* [arXiv:2212.08549](https://arxiv.org/abs/2212.08549)
- Hoffman & Gelman (2014). *The No-U-Turn Sampler.*
- Betancourt (2017). *A Conceptual Introduction to Hamiltonian Monte Carlo.*
- Gabrie et al. (2022). *Adaptive Monte Carlo augmented with normalizing flows.* [arXiv:2105.12603](https://arxiv.org/abs/2105.12603)
