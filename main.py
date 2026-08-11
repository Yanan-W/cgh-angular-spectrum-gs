# -*- coding: utf-8 -*-
"""
main.py
===============================================================================
Entry point for the Angular-Spectrum + Gerchberg-Saxton pure-phase CGH
simulation platform.

Run:
    python main.py

Produces (in ./figures/):
    dot_array_zero_init.png     - 4-panel result, 5x5 dot array target, zero-phase init
    dot_array_init_comparison.png
    vortex_beam_zero_init.png   - 4-panel result, vortex/doughnut target, zero-phase init
    vortex_beam_init_comparison.png

All physical simulation parameters live in the CONFIG dict directly below --
edit these to re-run at a different wavelength / resolution / propagation
distance without touching the algorithm code.
"""

import os

import numpy as np

from gs_algorithm import gs_algorithm
from propagation import angular_spectrum_propagate
from targets import generate_dot_array, generate_vortex_beam, generate_circular_aperture
from visualization import (
    set_publication_style,
    plot_result_panel,
    plot_init_comparison,
    plot_blas_comparison,
)

# =============================================================================
# CONFIGURATION -- single source of truth for all simulation parameters.
# =============================================================================
CONFIG = {
    # --- physical / optical parameters ---
    "wavelength": 532e-9,     # illumination wavelength, m (green laser)
    "N": 256,                 # grid size, N x N samples
    "dx": 8e-6,                # real-space sampling pitch, m (typical LCOS SLM pixel pitch)
    "z": 10e-3,                 # hologram -> target propagation distance, m
    "num_iterations": 60,     # GS iterations
    "use_band_limit": True,   # apply band-limited ASM (BLAS) frequency window
    "random_seed": 42,        # for the random-phase init mode, reproducibility

    # --- target-pattern parameters ---
    "dot_grid_size": 5,       # 5x5 dot array
    "vortex_topological_charge": 3,

    # --- BLAS aliasing demo parameters (independent of the GS experiments) ---
    "blas_demo_aperture_radius": 15 * 8e-6,  # hard-edged circular aperture, broadband source
    "blas_demo_z": 120e-3,                    # long propagation distance where aliasing bites

    # --- output ---
    "figures_dir": "figures",
}


def _print_header(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def run_target_experiment(target_name, target_intensity, config):
    """
    Run the full experiment for one target pattern: GS reconstruction with
    zero-phase init, GS reconstruction with random-phase init, and save both
    the 4-panel result figure and the init-comparison convergence figure.

    Parameters
    ----------
    target_name : str
        Human-readable + filename-safe target identifier, e.g. "dot_array".
    target_intensity : ndarray (N,N)
        Target intensity pattern to reconstruct.
    config : dict
        Global CONFIG dict (see module top).

    Returns
    -------
    result_zero, result_random : dict, dict
        Full gs_algorithm() outputs for each init strategy.
    """
    _print_header(f"Target: {target_name}")

    result_zero = gs_algorithm(
        target_intensity=target_intensity,
        wavelength=config["wavelength"],
        dx=config["dx"],
        z=config["z"],
        num_iterations=config["num_iterations"],
        init_mode="zero",
        use_band_limit=config["use_band_limit"],
    )
    print(f"[zero-init]   final MSE = {result_zero['mse_history'][-1]:.3e}")

    result_random = gs_algorithm(
        target_intensity=target_intensity,
        wavelength=config["wavelength"],
        dx=config["dx"],
        z=config["z"],
        num_iterations=config["num_iterations"],
        init_mode="random",
        use_band_limit=config["use_band_limit"],
        random_seed=config["random_seed"],
    )
    print(f"[random-init] final MSE = {result_random['mse_history'][-1]:.3e}")

    figures_dir = config["figures_dir"]

    panel_path = os.path.join(figures_dir, f"{target_name}_zero_init.png")
    plot_result_panel(
        target_intensity=target_intensity,
        phase_hologram=result_zero["phase_hologram"],
        reconstructed_intensity=result_zero["reconstructed_intensity"],
        mse_history=result_zero["mse_history"],
        dx=config["dx"],
        title=f"{target_name.replace('_', ' ').title()} -- GS reconstruction (zero-phase init)",
        save_path=panel_path,
    )
    print(f"  saved: {panel_path}")

    comparison_path = os.path.join(figures_dir, f"{target_name}_init_comparison.png")
    plot_init_comparison(
        mse_zero=result_zero["mse_history"],
        mse_random=result_random["mse_history"],
        title=f"{target_name.replace('_', ' ').title()} -- init strategy comparison",
        save_path=comparison_path,
    )
    print(f"  saved: {comparison_path}")

    return result_zero, result_random


def run_blas_aliasing_demo(config):
    """
    Propagate a hard-edged circular aperture over a long distance with and
    without the band-limited ASM (BLAS) frequency window, and save a
    side-by-side comparison figure. This is a standalone sanity check
    (independent of the GS experiments) that the BLAS filter required by
    this platform's physics spec is actually doing something: at long
    propagation distances relative to the aperture/window size, plain ASM
    develops visible spectral-aliasing artifacts (checkerboard energy
    replication across the frame) that BLAS suppresses.

    Parameters
    ----------
    config : dict
        Global CONFIG dict (see module top).
    """
    _print_header("BLAS aliasing demonstration")

    aperture_field = generate_circular_aperture(
        n_pixels=config["N"],
        dx=config["dx"],
        radius=config["blas_demo_aperture_radius"],
    )
    z_demo = config["blas_demo_z"]

    field_no_blas = angular_spectrum_propagate(
        aperture_field, config["wavelength"], config["dx"], z_demo, use_band_limit=False
    )
    field_blas = angular_spectrum_propagate(
        aperture_field, config["wavelength"], config["dx"], z_demo, use_band_limit=True
    )

    intensity_no_blas = np.abs(field_no_blas) ** 2
    intensity_blas = np.abs(field_blas) ** 2

    print(f"  propagation distance z = {z_demo * 1e3:.0f} mm (aperture radius = "
          f"{config['blas_demo_aperture_radius'] * 1e6:.0f} um)")
    print("  plain ASM shows checkerboard aliasing at this z:Lx ratio; BLAS removes it.")

    demo_path = os.path.join(config["figures_dir"], "blas_aliasing_demo.png")
    plot_blas_comparison(intensity_no_blas, intensity_blas, config["dx"], z_demo, save_path=demo_path)
    print(f"  saved: {demo_path}")


def main():
    config = CONFIG
    os.makedirs(config["figures_dir"], exist_ok=True)
    set_publication_style()

    window_mm = config["N"] * config["dx"] * 1e3
    print("CGH Simulation Platform -- Angular Spectrum Method + Gerchberg-Saxton")
    print(f"  wavelength      = {config['wavelength'] * 1e9:.0f} nm")
    print(f"  grid            = {config['N']} x {config['N']}  (dx = {config['dx'] * 1e6:.1f} um)")
    print(f"  field of view   = {window_mm:.3f} mm x {window_mm:.3f} mm")
    print(f"  propagation z   = {config['z'] * 1e3:.1f} mm")
    print(f"  GS iterations   = {config['num_iterations']}")
    print(f"  band-limited ASM= {config['use_band_limit']}")

    # --- Standalone sanity check: does BLAS actually remove aliasing? ---
    run_blas_aliasing_demo(config)

    # --- Target 1: dot array (multi-focus AR waveguide benchmark) ---
    dot_target = generate_dot_array(
        n_pixels=config["N"],
        dx=config["dx"],
        grid_size=config["dot_grid_size"],
    )
    run_target_experiment("dot_array", dot_target, config)

    # --- Target 2: vortex / doughnut beam (complex-shape / OAM benchmark) ---
    vortex_target = generate_vortex_beam(
        n_pixels=config["N"],
        dx=config["dx"],
        topological_charge=config["vortex_topological_charge"],
    )
    run_target_experiment("vortex_beam", vortex_target, config)

    _print_header("Done.")
    print(f"All figures written to ./{config['figures_dir']}/")


if __name__ == "__main__":
    main()
