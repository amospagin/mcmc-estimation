"""Kernel protocol — structural convention for MCMC transition kernels.

Every kernel module exposes three pure functions:

    init(position, logdensity_fn) -> KernelState
        Initialize kernel state from a starting position.

    build_kernel(logdensity_fn, **config) -> step_fn
        Close over the model and configuration to produce a step function.
        `config` is kernel-specific (step_size, inverse_mass_matrix, etc.).

    step_fn(rng_key, state: KernelState) -> (KernelState, KernelInfo)
        Perform one MCMC transition.  Must be JIT-compilable.

The separation between build_kernel and step_fn lets us JIT-compile the
inner loop while keeping configuration flexible.

Transform integration
---------------------
Kernels operate in *transformed* space.  The caller is responsible for
composing the transform with the log-density before passing it to
build_kernel:

    logdensity_in_z = lambda z: logdensity(transform.forward(z)) + transform.log_det_jac(z)
    step_fn = kernel.build_kernel(logdensity_in_z, ...)

This keeps kernels simple — they don't know about transforms.
"""
