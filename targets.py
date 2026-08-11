# -*- coding: utf-8 -*-
"""
targets.py
===============================================================================
Standard target light-field generators used to benchmark the GS phase
retrieval algorithm. Both targets are returned as *amplitude-normalized*
intensity patterns (max value = 1.0) in the observation (reconstruction)
plane, matching what a real intensity sensor / diffuser screen would record.

Included patterns
------------------
1. Dot array (`generate_dot_array`)
   An M x M grid of focused spots -- the canonical benchmark for
   multi-focal-spot generation used to validate AR waveguide out-coupler /
   eye-tracking illuminator style beam shaping.

2. Vortex / doughnut beam (`generate_vortex_beam`)
   A ring-shaped (doughnut) intensity pattern, i.e. the intensity signature
   of a Laguerre-Gaussian-like beam carrying orbital angular momentum (OAM)
   with topological charge l. Used to benchmark the algorithm's ability to
   recover complex, non-convex, high-spatial-frequency target shapes.

   Note on OAM: the classic (amplitude-only) GS loop reconstructed here only
   constrains *intensity* in the target plane, not phase, so it is not
   guaranteed to reproduce the l-fold phase singularity of a true vortex
   beam -- it simply learns *a* phase pattern whose |field|^2 matches the
   doughnut. This is still the standard way this pattern is used in the CGH
   literature as a "hard target shape" stress test. Genuinely imprinting the
   l*theta helical phase (e.g. for an OAM-carrying illuminator) is a
   different, direct-encoding design problem -- see the WGS/HIO extension
   points in `gs_algorithm.py` for how a phase-constrained variant could be
   layered on top of this.
"""

import numpy as np


def _centered_coordinate_grid(n_pixels, dx):
    """
    Build a centered real-space coordinate grid.

    Parameters
    ----------
    n_pixels : int
        Grid size N (square N x N grid).
    dx : float
        Real-space sampling pitch [m].

    Returns
    -------
    x_grid, y_grid : ndarray, shape (n_pixels, n_pixels)
        Coordinates [m], centered on the optical axis.
    """
    coords = (np.arange(n_pixels) - n_pixels // 2) * dx
    x_grid, y_grid = np.meshgrid(coords, coords, indexing="xy")
    return x_grid, y_grid


def generate_circular_aperture(n_pixels, dx, radius):
    """
    Generate a simple hard-edged circular aperture field (amplitude 1 inside
    radius, 0 outside). Not one of the two standard CGH benchmark targets --
    this is a diagnostic *source* field used to demonstrate the aliasing
    that band-limited ASM filtering (BLAS) removes at long propagation
    distances (see `main.run_blas_aliasing_demo`), since a hard circular
    edge is broadband in spatial frequency and makes aliasing artifacts
    especially visible.

    Parameters
    ----------
    n_pixels : int
        Grid size N (square N x N grid).
    dx : float
        Real-space sampling pitch [m].
    radius : float
        Aperture radius [m].

    Returns
    -------
    field : ndarray, complex, shape (n_pixels, n_pixels)
        Circular aperture field, amplitude in {0, 1}.
    """
    x_grid, y_grid = _centered_coordinate_grid(n_pixels, dx)
    r = np.sqrt(x_grid ** 2 + y_grid ** 2)
    field = (r < radius).astype(np.complex128)
    return field


def generate_dot_array(n_pixels, dx, grid_size=5, spot_sigma=None, pitch=None):
    """
    Generate an M x M array of focused Gaussian spots (target intensity),
    a standard multi-focus / beam-splitting benchmark (e.g. AR waveguide
    out-coupler grating validation, multi-spot structured illumination).

    Parameters
    ----------
    n_pixels : int
        Grid size N (square N x N grid).
    dx : float
        Real-space sampling pitch [m].
    grid_size : int, optional
        Number of spots per side, M (default 5 -> a 5x5 array).
    spot_sigma : float, optional
        Gaussian 1/e radius of each spot [m]. Defaults to 1.5 * dx.
    pitch : float, optional
        Center-to-center spot spacing [m]. Defaults to a value that spans
        ~60% of the simulation window, keeping spots away from the edges.

    Returns
    -------
    target_intensity : ndarray, shape (n_pixels, n_pixels)
        Amplitude-normalized (peak = 1.0) target intensity pattern.
    """
    if spot_sigma is None:
        spot_sigma = 1.5 * dx
    if pitch is None:
        window = n_pixels * dx
        pitch = 0.6 * window / (grid_size - 1) if grid_size > 1 else 0.0

    x_grid, y_grid = _centered_coordinate_grid(n_pixels, dx)

    offsets = (np.arange(grid_size) - (grid_size - 1) / 2.0) * pitch

    target_intensity = np.zeros((n_pixels, n_pixels), dtype=np.float64)
    for cy in offsets:
        for cx in offsets:
            target_intensity += np.exp(
                -((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2.0 * spot_sigma ** 2)
            )

    target_intensity /= target_intensity.max()
    return target_intensity


def generate_vortex_beam(n_pixels, dx, topological_charge=3, ring_radius=None, ring_width=None):
    """
    Generate a ring-shaped (doughnut) target intensity pattern, the intensity
    signature of an OAM-carrying vortex beam with topological charge l.

    The ring is modeled as a Gaussian annulus, exp[-2*(r - r0)^2 / w^2],
    centered at r0 = ring_radius with 1/e^2 half-width w = ring_width -- a
    good approximation to the |LG_{0,l}|^2 doughnut profile near its peak.
    Note that a vortex beam's *intensity* ring shape does not itself depend
    on the topological charge l (l only sets the helical phase, which this
    amplitude-only target does not encode -- see the module docstring);
    `topological_charge` is accepted and stored for bookkeeping / future
    phase-aware extensions rather than used to shape the intensity here.

    Parameters
    ----------
    n_pixels : int
        Grid size N (square N x N grid).
    dx : float
        Real-space sampling pitch [m].
    topological_charge : int, optional
        Topological charge l (default 3). Kept as metadata (see note above).
    ring_radius : float, optional
        Ring center radius r0 [m]. Defaults to 25% of the simulation window.
    ring_width : float, optional
        Ring 1/e^2 half-width w [m]. Defaults to ring_radius / 2.5.

    Returns
    -------
    target_intensity : ndarray, shape (n_pixels, n_pixels)
        Amplitude-normalized (peak = 1.0) target intensity pattern.
    """
    if ring_radius is None:
        ring_radius = 0.25 * n_pixels * dx
    if ring_width is None:
        ring_width = ring_radius / 2.5

    x_grid, y_grid = _centered_coordinate_grid(n_pixels, dx)
    r = np.sqrt(x_grid ** 2 + y_grid ** 2)

    target_intensity = np.exp(-2.0 * (r - ring_radius) ** 2 / ring_width ** 2)

    target_intensity /= target_intensity.max()
    return target_intensity
