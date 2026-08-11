# -*- coding: utf-8 -*-
"""
gs_algorithm.py
===============================================================================
Gerchberg-Saxton (GS) iterative phase-retrieval algorithm for pure-phase
computer-generated holography (CGH), built on top of the (band-limited)
Angular Spectrum propagator in `propagation.py`.

Algorithm outline
------------------
Let A_holo = 1 be the fixed amplitude enforced at the hologram (SLM) plane
(the hardware constraint of a *pure-phase* SLM -- it can only delay light,
not attenuate it) and A_target = sqrt(target_intensity) be the desired
target-plane amplitude. Starting from an initial phase guess phi_0, each GS
iteration performs one round-trip propagation with two amplitude
constraints enforced (and phase left free to evolve):

    1. U_holo        = A_holo * exp(j*phi_holo)                  [SLM constraint: |U_holo| = 1]
    2. U_target       = ASM_propagate(U_holo, +z)                 [forward propagation]
    3. phi_target     = angle(U_target)
    4. U_target'      = A_target * exp(j*phi_target)              [target-plane amplitude constraint]
    5. U_holo_new     = ASM_propagate(U_target', -z)               [backward propagation]
    6. phi_holo       = angle(U_holo_new)                          [discard amplitude -> pure-phase constraint]
    -> repeat from step 1.

Interview talking point -- why constant amplitude at the SLM?
    A phase-only LCOS/SLM physically cannot modulate amplitude pixel-by-
    pixel; every hologram pixel transmits/reflects the same optical power
    and only shifts phase. If step 6 kept |U_holo_new| instead of
    discarding it, the resulting hologram would require an amplitude mask
    that the target hardware simply does not have -- the field could not be
    displayed. Re-normalizing amplitude to 1 every iteration is therefore
    not a numerical convenience, it is what makes the output hologram
    physically realizable on a real pure-phase SLM.

Interview talking point -- why enforce the constraint only in intensity,
not phase, at the target plane?
    The eye / camera / photodetector at the target plane only measures
    intensity, so phase there is a "free" degree of freedom the algorithm
    is allowed to use to help convergence. Step 3-4 keeps whatever phase
    the propagation produced and only overwrites the amplitude to match
    the target -- this is exactly the Gerchberg-Saxton insight (1972) that
    lets an ill-posed intensity-only inverse problem become tractable by
    alternating projections between the two planes.
"""

import numpy as np

from propagation import angular_spectrum_propagate


def compute_mse(reconstructed_intensity, target_intensity):
    """
    Energy-normalized mean-squared error between a reconstructed and a
    target intensity pattern.

    Both patterns are first normalized to unit total energy (sum = 1)
    before differencing, so that the metric measures *shape* fidelity
    rather than being dominated by an arbitrary overall power scaling
    between the (energy non-conserving, phase-only) reconstruction and the
    target.

    Parameters
    ----------
    reconstructed_intensity : ndarray
        |U_target|^2 obtained by forward-propagating the current hologram.
    target_intensity : ndarray
        Desired target intensity pattern (same shape).

    Returns
    -------
    mse : float
        Mean-squared error between the two energy-normalized patterns.
    """
    recon_sum = reconstructed_intensity.sum()
    target_sum = target_intensity.sum()

    recon_norm = reconstructed_intensity / recon_sum if recon_sum > 0 else reconstructed_intensity
    target_norm = target_intensity / target_sum if target_sum > 0 else target_intensity

    return float(np.mean((recon_norm - target_norm) ** 2))


def apply_weighted_amplitude(current_amplitude, target_amplitude, weight_prev, iteration):
    """
    [Extension point -- not used by the classic GS loop below.]

    Weighted-GS (WGS) hook. Standard GS re-imposes the *raw* target
    amplitude A_target every iteration, which tends to plateau early and
    leave residual speckle noise / non-uniformity across the reconstructed
    spot pattern (particularly visible on the dot-array target). WGS
    instead reweights the enforced target amplitude per-pixel based on the
    running ratio between desired and achieved amplitude, e.g.:

        w_k(x,y)  = w_{k-1}(x,y) * A_target(x,y) / A_recon_{k-1}(x,y)
        A_target' = w_k(x,y) * A_target(x,y)

    which suppresses over/under-shooting pixels faster than uniform GS and
    typically improves both uniformity and diffraction efficiency.

    To wire this in: call this function in place of the flat
    `target_amplitude` re-imposition in `gs_algorithm`'s step 4, carrying
    `weight_prev` (initialized to ones) forward across iterations.

    Parameters
    ----------
    current_amplitude : ndarray
        |U_target| achieved on this iteration, before re-imposing weights.
    target_amplitude : ndarray
        Desired (unweighted) target amplitude, sqrt(target_intensity).
    weight_prev : ndarray
        Per-pixel weight map carried over from the previous iteration.
    iteration : int
        Current iteration index (some WGS variants ramp weighting in
        gradually rather than from iteration 0).

    Returns
    -------
    NotImplementedError
        This is a documented extension point, intentionally unimplemented.
    """
    raise NotImplementedError(
        "WGS extension point -- see docstring for the intended weighted "
        "target-amplitude update rule."
    )


def apply_hio_feedback(hologram_field_new, hologram_field_prev, feedback_beta=0.9):
    """
    [Extension point -- not used by the classic GS loop below.]

    Fienup Hybrid Input-Output (HIO) hook. Plain GS can stagnate in local
    minima (stalled MSE plateaus, visible stagnation in the convergence
    curve produced by this platform). HIO avoids simply overwriting the
    hologram-plane estimate each iteration; instead, outside the
    constraint region it feeds back a damped correction:

        u_holo_{k+1} = g_k                              inside the SLM constraint region
        u_holo_{k+1} = u_holo_k - beta * g_k             outside it

    where g_k is the raw (unconstrained) inverse-propagated estimate. For
    the pure-phase-SLM case here, "inside the constraint" naturally
    corresponds to pixels already satisfying |U| = 1; the correction term
    prevents the algorithm from repeatedly re-committing to the same local
    optimum.

    To wire this in: call this function instead of directly overwriting
    `phase_holo` with `angle(U_holo_new)` in `gs_algorithm`'s step 6,
    passing the previous iteration's hologram field alongside the new one.

    Parameters
    ----------
    hologram_field_new : ndarray, complex
        Raw (unconstrained) back-propagated hologram-plane field estimate
        for this iteration.
    hologram_field_prev : ndarray, complex
        Hologram-plane field actually used on the previous iteration.
    feedback_beta : float, optional
        HIO feedback gain, typically in [0.5, 1.0]. Default 0.9.

    Returns
    -------
    NotImplementedError
        This is a documented extension point, intentionally unimplemented.
    """
    raise NotImplementedError(
        "HIO extension point -- see docstring for the intended damped "
        "feedback update rule."
    )


def gs_algorithm(
    target_intensity,
    wavelength,
    dx,
    z,
    num_iterations=50,
    init_mode="zero",
    use_band_limit=True,
    random_seed=None,
):
    """
    Run the Gerchberg-Saxton iterative phase-retrieval algorithm to design a
    pure-phase hologram that reconstructs `target_intensity` at distance z.

    Parameters
    ----------
    target_intensity : ndarray, shape (N, N)
        Desired target-plane intensity pattern (need not be normalized;
        it is amplitude-normalized internally via sqrt()).
    wavelength : float
        Illumination wavelength [m].
    dx : float
        Real-space sampling pitch [m].
    z : float
        Hologram-to-target propagation distance [m] (> 0).
    num_iterations : int, optional
        Number of GS iterations to run (default 50).
    init_mode : {'zero', 'random'}, optional
        Initial hologram-plane phase guess:
        - 'zero'   : uniform phi_0 = 0 everywhere (deterministic baseline).
        - 'random' : phi_0 ~ Uniform(-pi, pi) per pixel (helps escape the
                     symmetric local optimum that a flat phase start can
                     fall into, at the cost of a noisier early hologram).
    use_band_limit : bool, optional
        Whether the underlying ASM propagator applies the BLAS frequency
        window (default True; strongly recommended -- see propagation.py).
    random_seed : int, optional
        Seed for the 'random' init mode, for reproducibility.

    Returns
    -------
    result : dict with keys
        'phase_hologram'         : ndarray (N,N) -- final SLM phase map, wrapped to (-pi, pi].
        'reconstructed_intensity': ndarray (N,N) -- |forward-propagated hologram|^2 at distance z.
        'mse_history'            : ndarray (num_iterations,) -- MSE after each iteration.
        'target_amplitude'       : ndarray (N,N) -- sqrt(target_intensity), for reference/plotting.
    """
    if z <= 0:
        raise ValueError("Propagation distance z must be positive.")

    n_pixels = target_intensity.shape[0]
    target_amplitude = np.sqrt(np.clip(target_intensity, 0.0, None))

    # --- initial hologram-plane phase guess ---
    if init_mode == "zero":
        phase_holo = np.zeros((n_pixels, n_pixels), dtype=np.float64)
    elif init_mode == "random":
        rng = np.random.default_rng(random_seed)
        phase_holo = rng.uniform(-np.pi, np.pi, size=(n_pixels, n_pixels))
    else:
        raise ValueError("init_mode must be 'zero' or 'random', got %r" % (init_mode,))

    mse_history = np.zeros(num_iterations, dtype=np.float64)
    reconstructed_intensity = None

    for iteration in range(num_iterations):
        # 1. Pure-phase SLM constraint: amplitude fixed at 1, only phase carries information.
        holo_field = np.exp(1j * phase_holo)

        # 2. Forward propagation, hologram plane -> target plane.
        target_field = angular_spectrum_propagate(holo_field, wavelength, dx, z, use_band_limit)

        reconstructed_intensity = np.abs(target_field) ** 2
        mse_history[iteration] = compute_mse(reconstructed_intensity, target_intensity)

        # 3-4. Target-plane amplitude constraint: keep the propagated phase,
        #      overwrite amplitude with the desired target amplitude. Phase
        #      at the (intensity-only) target plane is a free parameter the
        #      algorithm is allowed to exploit -- this is the core GS trick.
        target_phase = np.angle(target_field)
        constrained_target_field = target_amplitude * np.exp(1j * target_phase)

        # 5. Backward propagation, target plane -> hologram plane (z -> -z).
        holo_field_new = angular_spectrum_propagate(
            constrained_target_field, wavelength, dx, -z, use_band_limit
        )

        # 6. Pure-phase SLM constraint again: discard the returned amplitude,
        #    keep only phase, ready for the next round-trip.
        phase_holo = np.angle(holo_field_new)

    result = {
        "phase_hologram": phase_holo,
        "reconstructed_intensity": reconstructed_intensity,
        "mse_history": mse_history,
        "target_amplitude": target_amplitude,
    }
    return result
