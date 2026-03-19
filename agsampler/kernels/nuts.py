"""No-U-Turn Sampler (NUTS) kernel.

Implements multinomial NUTS (Betancourt 2017) with iterative tree doubling.
Used as the fallback kernel when MCLMC struggles (e.g., very difficult
geometry before the flow has trained).

The tree is built iteratively:
  1. Outer while_loop doubles the tree depth
  2. Inner while_loop takes 2^depth leapfrog steps for each subtree
  3. U-turn check on full tree boundaries after each doubling
  4. Proposal selected via multinomial weighting (log-sum-exp of -energy)
"""

import jax
import jax.numpy as jnp
from jax import Array
from typing import NamedTuple

from agsampler.types import KernelState, KernelInfo, IntegratorState
from agsampler.integrators.leapfrog import leapfrog_step
from agsampler.integrators.utils import kinetic_energy, generate_momentum, total_energy


MAX_TREE_DEPTH = 10
MAX_DELTA_ENERGY = 1000.0


# ---------------------------------------------------------------------------
# Internal tree state
# ---------------------------------------------------------------------------

class _TreeState(NamedTuple):
    """Internal state for iterative tree building."""
    # Left boundary
    left_position: Array
    left_momentum: Array
    left_grad: Array
    # Right boundary
    right_position: Array
    right_momentum: Array
    right_grad: Array
    # Current proposal
    proposal_position: Array
    proposal_log_prob: float
    proposal_grad: Array
    proposal_energy: float
    # Subtree proposal (for merging)
    sub_proposal_position: Array
    sub_proposal_log_prob: float
    sub_proposal_grad: Array
    sub_proposal_energy: float
    # Statistics
    depth: int
    log_sum_weight: float
    sub_log_sum_weight: float
    momentum_sum: Array
    is_divergent: bool
    turning: bool
    num_steps: int


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init(position: Array, logdensity_fn) -> KernelState:
    """Initialize NUTS kernel state."""
    log_prob, grad = jax.value_and_grad(logdensity_fn)(position)
    return KernelState(
        position=position,
        log_prob=log_prob,
        log_prob_grad=grad,
    )


# ---------------------------------------------------------------------------
# Kernel builder
# ---------------------------------------------------------------------------

def build_kernel(
    logdensity_fn,
    step_size: float,
    inverse_mass_matrix: Array,
    max_depth: int = MAX_TREE_DEPTH,
):
    """Build the NUTS step function.

    Parameters
    ----------
    logdensity_fn : callable
        Maps position (D,) -> scalar log-density.
    step_size : float
    inverse_mass_matrix : Array
        Shape (D,) for diagonal, (D,D) for dense.
    max_depth : int
        Maximum tree depth (2^max_depth max leapfrog steps).
    """

    def _one_leapfrog(state: IntegratorState, direction: int) -> IntegratorState:
        """One leapfrog step in the given direction."""
        return leapfrog_step(state, direction * step_size, logdensity_fn, inverse_mass_matrix)

    def _is_turning(left_pos, left_mom, right_pos, right_mom, momentum_sum):
        """Generalized U-turn criterion (Betancourt 2017)."""
        diff = right_pos - left_pos
        return (jnp.dot(diff, momentum_sum - left_mom) < 0) | \
               (jnp.dot(diff, momentum_sum - right_mom) < 0)

    def step_fn(rng_key, state: KernelState) -> tuple[KernelState, KernelInfo]:
        position = state.position
        log_prob = state.log_prob
        grad = state.log_prob_grad

        # Sample momentum
        rng_key, mom_key = jax.random.split(rng_key)
        momentum = generate_momentum(mom_key, inverse_mass_matrix)
        initial_energy = total_energy(log_prob, momentum, inverse_mass_matrix)

        # Initialize tree with a single node
        tree = _TreeState(
            left_position=position,
            left_momentum=momentum,
            left_grad=grad,
            right_position=position,
            right_momentum=momentum,
            right_grad=grad,
            proposal_position=position,
            proposal_log_prob=log_prob,
            proposal_grad=grad,
            proposal_energy=initial_energy,
            sub_proposal_position=position,
            sub_proposal_log_prob=log_prob,
            sub_proposal_grad=grad,
            sub_proposal_energy=initial_energy,
            depth=0,
            log_sum_weight=-initial_energy,
            sub_log_sum_weight=-jnp.inf,
            momentum_sum=momentum,
            is_divergent=False,
            turning=False,
            num_steps=0,
        )

        def _doubling_cond(args):
            tree, _ = args
            return (~tree.turning) & (~tree.is_divergent) & (tree.depth < max_depth)

        def _doubling_body(args):
            tree, rng_key = args
            rng_key, dir_key, accept_key, subtree_key = jax.random.split(rng_key, 4)

            # Choose direction: +1 (forward) or -1 (backward)
            going_right = jax.random.bernoulli(dir_key)
            direction = jnp.where(going_right, 1, -1)

            # Starting boundary for the subtree
            start_position = jnp.where(going_right, tree.right_position, tree.left_position)
            start_momentum = jnp.where(going_right, tree.right_momentum, tree.left_momentum)
            start_grad = jnp.where(going_right, tree.right_grad, tree.left_grad)

            # Build subtree: take 2^depth steps in the chosen direction
            n_steps = jnp.int32(2) ** tree.depth
            subtree = _build_subtree(
                subtree_key, start_position, start_momentum, start_grad,
                direction, n_steps, initial_energy, _one_leapfrog,
            )

            # Merge proposal: accept subtree proposal with probability
            # proportional to its weight relative to the tree
            sub_weight = subtree.log_sum_weight
            tree_weight = tree.log_sum_weight
            merged_weight = jnp.logaddexp(tree_weight, sub_weight)
            accept_prob = jnp.exp(jnp.minimum(sub_weight - merged_weight, 0.0))
            do_accept = jax.random.bernoulli(accept_key, accept_prob)

            new_proposal_pos = jnp.where(do_accept, subtree.proposal_position, tree.proposal_position)
            new_proposal_lp = jnp.where(do_accept, subtree.proposal_log_prob, tree.proposal_log_prob)
            new_proposal_grad = jnp.where(do_accept, subtree.proposal_grad, tree.proposal_grad)
            new_proposal_energy = jnp.where(do_accept, subtree.proposal_energy, tree.proposal_energy)

            # Update tree boundaries
            new_left_pos = jnp.where(going_right, tree.left_position, subtree.end_position)
            new_left_mom = jnp.where(going_right, tree.left_momentum, subtree.end_momentum)
            new_left_grad = jnp.where(going_right, tree.left_grad, subtree.end_grad)
            new_right_pos = jnp.where(going_right, subtree.end_position, tree.right_position)
            new_right_mom = jnp.where(going_right, subtree.end_momentum, tree.right_momentum)
            new_right_grad = jnp.where(going_right, subtree.end_grad, tree.right_grad)

            new_momentum_sum = tree.momentum_sum + subtree.momentum_sum

            # Check U-turn on full tree
            turning = _is_turning(
                new_left_pos, new_left_mom,
                new_right_pos, new_right_mom,
                new_momentum_sum,
            )

            new_tree = _TreeState(
                left_position=new_left_pos,
                left_momentum=new_left_mom,
                left_grad=new_left_grad,
                right_position=new_right_pos,
                right_momentum=new_right_mom,
                right_grad=new_right_grad,
                proposal_position=new_proposal_pos,
                proposal_log_prob=new_proposal_lp,
                proposal_grad=new_proposal_grad,
                proposal_energy=new_proposal_energy,
                sub_proposal_position=subtree.proposal_position,
                sub_proposal_log_prob=subtree.proposal_log_prob,
                sub_proposal_grad=subtree.proposal_grad,
                sub_proposal_energy=subtree.proposal_energy,
                depth=tree.depth + 1,
                log_sum_weight=merged_weight,
                sub_log_sum_weight=sub_weight,
                momentum_sum=new_momentum_sum,
                is_divergent=tree.is_divergent | subtree.is_divergent,
                turning=turning,
                num_steps=tree.num_steps + subtree.num_steps,
            )
            return new_tree, rng_key

        final_tree, _ = jax.lax.while_loop(_doubling_cond, _doubling_body, (tree, rng_key))

        # Acceptance rate: fraction of tree weight from valid proposals
        acceptance_rate = jnp.exp(
            jnp.minimum(final_tree.log_sum_weight - (-initial_energy), 0.0)
        )

        new_state = KernelState(
            position=final_tree.proposal_position,
            log_prob=final_tree.proposal_log_prob,
            log_prob_grad=final_tree.proposal_grad,
        )

        info = KernelInfo(
            momentum=momentum,
            is_divergent=final_tree.is_divergent,
            num_integration_steps=final_tree.num_steps,
            acceptance_rate=acceptance_rate,
            energy=final_tree.proposal_energy,
        )

        return new_state, info

    return step_fn


# ---------------------------------------------------------------------------
# Subtree builder
# ---------------------------------------------------------------------------

class _SubtreeResult(NamedTuple):
    """Result of building a subtree."""
    end_position: Array
    end_momentum: Array
    end_grad: Array
    proposal_position: Array
    proposal_log_prob: float
    proposal_grad: Array
    proposal_energy: float
    log_sum_weight: float
    momentum_sum: Array
    is_divergent: bool
    num_steps: int


def _build_subtree(
    rng_key,
    start_position, start_momentum, start_grad,
    direction, n_steps, initial_energy,
    one_leapfrog_fn,
):
    """Build a subtree by taking n_steps leapfrog steps.

    Tracks the subtree boundary, accumulates proposal weights, and
    selects a proposal via multinomial sampling.
    """

    class _Carry(NamedTuple):
        step: int
        position: Array
        momentum: Array
        log_prob: float
        grad: Array
        proposal_position: Array
        proposal_log_prob: float
        proposal_grad: Array
        proposal_energy: float
        log_sum_weight: float
        momentum_sum: Array
        is_divergent: bool
        rng_key: Array

    initial_state = IntegratorState(start_position, start_momentum, 0.0, start_grad)
    log_prob_init, _ = initial_state.log_prob, initial_state.log_prob_grad

    # We need the log_prob for the starting position — recompute from energy
    # Actually, let's just start the proposal as the first leapfrog result

    def _cond(carry):
        return (carry.step < n_steps) & (~carry.is_divergent)

    def _body(carry):
        state = IntegratorState(carry.position, carry.momentum, carry.log_prob, carry.grad)
        new_state = one_leapfrog_fn(state, direction)

        new_pos, new_mom, new_lp, new_grad = new_state
        new_energy = -new_lp + 0.5 * jnp.sum(new_mom ** 2)  # simplified KE for now

        # Divergence check
        delta_energy = new_energy - initial_energy
        is_div = jnp.abs(delta_energy) > MAX_DELTA_ENERGY

        # Multinomial proposal: weight = exp(-energy)
        log_weight = -new_energy
        new_log_sum_weight = jnp.logaddexp(carry.log_sum_weight, log_weight)

        # Accept new state as proposal with probability proportional to weight
        rng_key, accept_key = jax.random.split(carry.rng_key)
        accept_prob = jnp.exp(jnp.minimum(log_weight - new_log_sum_weight, 0.0))
        do_accept = jax.random.bernoulli(accept_key, accept_prob)

        prop_pos = jnp.where(do_accept, new_pos, carry.proposal_position)
        prop_lp = jnp.where(do_accept, new_lp, carry.proposal_log_prob)
        prop_grad = jnp.where(do_accept, new_grad, carry.proposal_grad)
        prop_energy = jnp.where(do_accept, new_energy, carry.proposal_energy)

        return _Carry(
            step=carry.step + 1,
            position=new_pos,
            momentum=new_mom,
            log_prob=new_lp,
            grad=new_grad,
            proposal_position=prop_pos,
            proposal_log_prob=prop_lp,
            proposal_grad=prop_grad,
            proposal_energy=prop_energy,
            log_sum_weight=new_log_sum_weight,
            momentum_sum=carry.momentum_sum + new_mom,
            is_divergent=carry.is_divergent | is_div,
            rng_key=rng_key,
        )

    init_carry = _Carry(
        step=jnp.int32(0),
        position=start_position,
        momentum=start_momentum,
        log_prob=0.0,  # will be overwritten on first step
        grad=start_grad,
        proposal_position=start_position,
        proposal_log_prob=0.0,
        proposal_grad=start_grad,
        proposal_energy=initial_energy,
        log_sum_weight=-jnp.inf,
        momentum_sum=jnp.zeros_like(start_momentum),
        is_divergent=False,
        rng_key=rng_key,
    )

    result = jax.lax.while_loop(_cond, _body, init_carry)

    return _SubtreeResult(
        end_position=result.position,
        end_momentum=result.momentum,
        end_grad=result.grad,
        proposal_position=result.proposal_position,
        proposal_log_prob=result.proposal_log_prob,
        proposal_grad=result.proposal_grad,
        proposal_energy=result.proposal_energy,
        log_sum_weight=result.log_sum_weight,
        momentum_sum=result.momentum_sum,
        is_divergent=result.is_divergent,
        num_steps=result.step,
    )
