# -*- coding: utf-8 -*-
"""
visualization.py
===============================================================================
Publication-style plotting utilities for the CGH simulation platform.

Two entry points are used by main.py:
    - set_publication_style() : call once, configures global matplotlib rcParams.
    - plot_result_panel()     : 2x2 summary panel (target / hologram / recon / MSE curve).
    - plot_init_comparison()  : zero-init vs random-init MSE convergence overlay.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def set_publication_style():
    """
    Configure global matplotlib rcParams for consistent, journal-style
    figures (serif font, DPI >= 300, clean axis defaults) across every
    figure produced by this platform.
    """
    matplotlib.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def _add_scale_bar(ax, n_pixels, dx, fraction=0.25, color="white"):
    """
    Draw a physical-length scale bar in the bottom-left corner of an image
    axis, in place of numeric tick labels (kept off for a clean CGH-style
    intensity/phase map).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis the image was drawn on with imshow (pixel-index extent assumed).
    n_pixels : int
        Grid size N (image assumed square, N x N).
    dx : float
        Real-space sampling pitch [m], used to convert the bar length to mm.
    fraction : float, optional
        Target scale-bar length as a fraction of the field of view.
    color : str, optional
        Scale-bar / label color (default 'white', for dark colormaps).
    """
    window_mm = n_pixels * dx * 1e3  # full field of view, in mm
    bar_mm = round(fraction * window_mm, 1) or 0.1
    bar_px = bar_mm / (dx * 1e3)

    margin = 0.06 * n_pixels
    y0 = n_pixels - margin
    x0 = margin

    ax.plot([x0, x0 + bar_px], [y0, y0], color=color, linewidth=2.5, solid_capstyle="butt")
    ax.text(
        x0 + bar_px / 2.0,
        y0 - margin * 0.5,
        f"{bar_mm:g} mm",
        color=color,
        ha="center",
        va="bottom",
        fontsize=8,
    )


def plot_result_panel(
    target_intensity,
    phase_hologram,
    reconstructed_intensity,
    mse_history,
    dx,
    title=None,
    save_path=None,
):
    """
    Render the standard four-panel CGH results figure:
        (a) target intensity distribution
        (b) computed hologram phase (SLM loading pattern)
        (c) reconstructed intensity distribution
        (d) MSE convergence curve vs. iteration

    Parameters
    ----------
    target_intensity : ndarray (N,N)
        Desired target-plane intensity.
    phase_hologram : ndarray (N,N)
        Final SLM phase map, radians, wrapped to (-pi, pi].
    reconstructed_intensity : ndarray (N,N)
        Forward-propagated |hologram field|^2 at the target plane.
    mse_history : ndarray (num_iterations,)
        Per-iteration MSE, as returned by gs_algorithm().
    dx : float
        Real-space sampling pitch [m], used for the scale bars.
    title : str, optional
        Overall figure title (e.g. target name).
    save_path : str, optional
        If given, the figure is saved to this path (dpi >= 300, per
        set_publication_style()).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_pixels = target_intensity.shape[0]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.2))

    # (a) target intensity
    ax = axes[0, 0]
    im = ax.imshow(target_intensity, cmap="inferno")
    ax.set_title("(a) Target intensity")
    ax.axis("off")
    _add_scale_bar(ax, n_pixels, dx)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="norm. intensity")

    # (b) hologram phase (SLM loading pattern)
    ax = axes[0, 1]
    im = ax.imshow(phase_hologram, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    ax.set_title("(b) Hologram phase (SLM map)")
    ax.axis("off")
    _add_scale_bar(ax, n_pixels, dx, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="phase [rad]")
    cbar.set_ticks([-np.pi, 0, np.pi])
    cbar.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])

    # (c) reconstructed intensity
    # Displayed peak-normalized (0-1), matching panel (a)'s scale, purely for
    # visual shape comparison. The raw reconstructed intensity is naturally
    # much brighter than the target (coherent energy concentrated from the
    # whole aperture into small spots/rings) -- physically correct, but not
    # directly comparable on a shared color scale without this normalization.
    # The MSE metric itself is computed on energy-normalized patterns
    # upstream in gs_algorithm.compute_mse() and is unaffected by this.
    ax = axes[1, 0]
    recon_peak = reconstructed_intensity.max()
    recon_display = reconstructed_intensity / recon_peak if recon_peak > 0 else reconstructed_intensity
    im = ax.imshow(recon_display, cmap="inferno", vmin=0, vmax=1)
    ax.set_title("(c) Reconstructed intensity (peak-norm.)")
    ax.axis("off")
    _add_scale_bar(ax, n_pixels, dx)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="norm. intensity")

    # (d) MSE convergence curve
    ax = axes[1, 1]
    iterations = np.arange(1, len(mse_history) + 1)
    ax.plot(iterations, mse_history, color="firebrick", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("GS iteration")
    ax.set_ylabel("MSE (energy-normalized)")
    ax.set_title("(d) Convergence curve")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

    if title:
        fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97) if title else (0, 0, 1, 1))

    if save_path:
        fig.savefig(save_path)

    return fig


def plot_blas_comparison(intensity_no_blas, intensity_blas, dx, z, save_path=None):
    """
    Side-by-side (log-scale) comparison of a propagated intensity pattern
    computed WITHOUT vs. WITH the band-limited ASM (BLAS) frequency window,
    demonstrating the spectral-aliasing artifacts BLAS is designed to
    remove at long propagation distances / coarse sampling (see
    propagation.py module docstring and Matsushima & Shimobaba 2009).

    Parameters
    ----------
    intensity_no_blas : ndarray (N,N)
        |field|^2 propagated with use_band_limit=False.
    intensity_blas : ndarray (N,N)
        |field|^2 propagated with use_band_limit=True, same source/z.
    dx : float
        Real-space sampling pitch [m], for the scale bars.
    z : float
        Propagation distance used [m], shown in the panel titles.
    save_path : str, optional
        If given, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_pixels = intensity_no_blas.shape[0]
    eps_no = 1e-6 * intensity_no_blas.max()
    eps_bl = 1e-6 * intensity_blas.max()

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4))

    ax = axes[0]
    im = ax.imshow(np.log10(intensity_no_blas + eps_no), cmap="inferno")
    ax.set_title(f"Plain ASM (no BLAS), z = {z * 1e3:.0f} mm\naliasing artifacts")
    ax.axis("off")
    _add_scale_bar(ax, n_pixels, dx)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=r"$\log_{10}$(intensity)")

    ax = axes[1]
    im = ax.imshow(np.log10(intensity_blas + eps_bl), cmap="inferno")
    ax.set_title(f"Band-limited ASM, z = {z * 1e3:.0f} mm\nartifacts suppressed")
    ax.axis("off")
    _add_scale_bar(ax, n_pixels, dx)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=r"$\log_{10}$(intensity)")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig


def plot_init_comparison(mse_zero, mse_random, title=None, save_path=None):
    """
    Overlay MSE convergence curves for zero-phase vs. random-phase GS
    initialization, to compare their convergence behavior.

    Parameters
    ----------
    mse_zero : ndarray (num_iterations,)
        MSE history from a run with init_mode='zero'.
    mse_random : ndarray (num_iterations,)
        MSE history from a run with init_mode='random' (same target/params).
    title : str, optional
        Figure title (e.g. target name).
    save_path : str, optional
        If given, the figure is saved to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    iterations = np.arange(1, len(mse_zero) + 1)

    ax.plot(iterations, mse_zero, color="steelblue", linewidth=1.6, label="Zero-phase init")
    ax.plot(iterations, mse_random, color="darkorange", linewidth=1.6, label="Random-phase init")
    ax.set_yscale("log")
    ax.set_xlabel("GS iteration")
    ax.set_ylabel("MSE (energy-normalized)")
    ax.set_title(title or "Initialization strategy comparison")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig
