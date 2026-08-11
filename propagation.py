# -*- coding: utf-8 -*-
"""
propagation.py
===============================================================================
Scalar diffraction propagation via the Angular Spectrum Method (ASM).

Physics
-------
Given a complex field U(x, y; 0) in the source plane, the angular spectrum
(2-D spatial Fourier transform) is

    A(fx, fy; 0) = F{ U(x, y; 0) }

Each plane-wave component propagates independently over a distance z by
accumulating a phase set by the free-space transfer function:

    H(fx, fy; z) = exp[ j * 2*pi * z * sqrt( 1/lambda**2 - fx**2 - fy**2 ) ]   (propagating, fx^2+fy^2 < 1/lambda^2)
                  = 0                                                          (evanescent, fx^2+fy^2 >= 1/lambda^2)

and the field at distance z is recovered by the inverse transform:

    U(x, y; z) = F^-1{ A(fx, fy; 0) * H(fx, fy; z) }

This is the *exact* Rayleigh-Sommerfeld / angular-spectrum result -- no
paraxial (Fresnel) approximation of the square root is made, so the model
stays valid for large diffraction angles / short propagation distances.

Two numerical safeguards are mandatory for a physically-trustworthy CGH
simulator and are both implemented below:

1. Evanescent-wave cutoff
   Spatial frequencies with fx^2 + fy^2 >= 1/lambda^2 correspond to
   inhomogeneous (evanescent) waves that decay exponentially and carry no
   energy to the far field. Numerically, sqrt(1/lambda^2 - fx^2 - fy^2)
   would otherwise become imaginary, turning H into an exponentially
   *growing* factor -- a classic source of numerical blow-up. We therefore
   hard-zero H outside the propagating-wave circle.

2. Band-limited angular spectrum (BLAS) filtering
   Even after removing evanescent components, H(fx,fy;z) is a chirp whose
   local spatial frequency grows with z. When the simulation window is
   finite (size Lx = N*dx), that chirp's *local frequency* can exceed the
   Nyquist frequency of the discrete grid well before reaching the
   fx^2+fy^2 = 1/lambda^2 boundary -- causing spectral aliasing that shows
   up as spurious high-frequency artifacts / energy leakage, especially at
   long propagation distances or coarse sampling. Following Matsushima &
   Shimobaba (Opt. Express 17, 19662, 2009), the transfer function is
   additionally windowed to the band limit

       fx_limit = 1 / ( lambda * sqrt( (2*z/Lx)**2 + 1 ) )
       fy_limit = 1 / ( lambda * sqrt( (2*z/Ly)**2 + 1 ) )

   i.e. |fx| < fx_limit and |fy| < fy_limit, zero elsewhere.

References
----------
- Goodman, J. W., "Introduction to Fourier Optics", 3rd ed., Ch. 3.
- Matsushima, K.; Shimobaba, T. "Band-Limited Angular Spectrum Method for
  Numerical Simulation of Free-Space Propagation in Far and Near Fields."
  Optics Express 17(22), 19662-19673 (2009).
"""

import numpy as np


def _spatial_frequency_grids(n_pixels, dx):
    """
    Build the 2-D spatial-frequency coordinate grids matching numpy's FFT
    ordering (i.e. unshifted, DC at index 0), so they can be multiplied
    directly against np.fft.fft2 output without an fftshift round-trip.

    Parameters
    ----------
    n_pixels : int
        Number of samples per side of the (square) simulation grid.
    dx : float
        Real-space sampling pitch [m].

    Returns
    -------
    fx_grid, fy_grid : ndarray, shape (n_pixels, n_pixels)
        Spatial frequency coordinates [1/m], in FFT (unshifted) order.
    """
    fx = np.fft.fftfreq(n_pixels, d=dx)  # cycles / m
    fx_grid, fy_grid = np.meshgrid(fx, fx, indexing="xy")
    return fx_grid, fy_grid


def angular_spectrum_transfer_function(n_pixels, dx, wavelength, z, use_band_limit=True):
    """
    Build the (band-limited) angular-spectrum transfer function H(fx, fy; z).

    Parameters
    ----------
    n_pixels : int
        Grid size N (square N x N grid assumed).
    dx : float
        Real-space sampling pitch [m].
    wavelength : float
        Illumination wavelength [m].
    z : float
        Signed propagation distance [m]. z > 0 propagates forward
        (source -> observation), z < 0 propagates backward
        (observation -> source); the physics is symmetric in z.
    use_band_limit : bool, optional
        If True (default), apply the Matsushima band-limited ASM (BLAS)
        window on top of the evanescent cutoff. Should normally stay True;
        disabling it is only useful to visualise the aliasing BLAS is
        designed to remove.

    Returns
    -------
    H : ndarray, complex, shape (n_pixels, n_pixels)
        Transfer function in FFT (unshifted) frequency order, ready to
        multiply element-wise against np.fft.fft2(field).
    """
    fx, fy = _spatial_frequency_grids(n_pixels, dx)

    # --- propagating-wave radicand: kz/k = sqrt(1/lambda^2 - fx^2 - fy^2) ---
    radicand = (1.0 / wavelength) ** 2 - fx ** 2 - fy ** 2

    # Evanescent-wave cutoff: fx^2 + fy^2 >= 1/lambda^2 carries no
    # propagating energy -> force H = 0 there instead of letting the sqrt
    # go imaginary (which would otherwise produce an exponentially
    # *growing*, numerically unstable term exp(+|...| ) for z > 0).
    propagating_mask = radicand > 0.0
    kz_over_k = np.zeros_like(radicand)
    kz_over_k[propagating_mask] = np.sqrt(radicand[propagating_mask])

    H = np.exp(1j * 2.0 * np.pi * z * kz_over_k) * propagating_mask

    if use_band_limit:
        # --- Band-Limited Angular Spectrum (BLAS) window, Matsushima 2009 ---
        Lx = n_pixels * dx
        Ly = n_pixels * dx  # square grid
        abs_z = abs(z) if z != 0.0 else 1e-12  # avoid /0 for the trivial z=0 case

        fx_limit = 1.0 / (wavelength * np.sqrt((2.0 * abs_z / Lx) ** 2 + 1.0))
        fy_limit = 1.0 / (wavelength * np.sqrt((2.0 * abs_z / Ly) ** 2 + 1.0))

        band_mask = (np.abs(fx) < fx_limit) & (np.abs(fy) < fy_limit)
        H = H * band_mask

    return H


def angular_spectrum_propagate(field, wavelength, dx, z, use_band_limit=True):
    """
    Propagate a complex scalar field by distance z using the (band-limited)
    Angular Spectrum Method.

    Parameters
    ----------
    field : ndarray, complex, shape (N, N)
        Input complex field U(x, y; 0).
    wavelength : float
        Illumination wavelength [m].
    dx : float
        Real-space sampling pitch [m] (assumed equal along x and y).
    z : float
        Signed propagation distance [m]. Use a negative z to propagate the
        field backward (this is exactly how the GS algorithm below
        back-propagates from the target plane to the hologram plane).
    use_band_limit : bool, optional
        Whether to apply the BLAS frequency window (default True; see
        `angular_spectrum_transfer_function`).

    Returns
    -------
    propagated_field : ndarray, complex, shape (N, N)
        U(x, y; z).
    """
    n_pixels = field.shape[0]
    H = angular_spectrum_transfer_function(n_pixels, dx, wavelength, z, use_band_limit)

    # F{U(x,y;0)} -> * H(fx,fy;z) -> F^-1{...} = U(x,y;z)
    spectrum = np.fft.fft2(field)
    propagated_spectrum = spectrum * H
    propagated_field = np.fft.ifft2(propagated_spectrum)
    return propagated_field
