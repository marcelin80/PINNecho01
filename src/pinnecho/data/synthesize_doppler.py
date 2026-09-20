"""Synthesise Doppler-like measurements from a ground-truth velocity field.

Turns a dense ground-truth velocity field (from IBFE output, or the Stage-1
synthetic LV) into the sparse, single-component, noisy measurements a colour /
spectral Doppler acquisition would produce from one or more apical windows. Only
the component of velocity *along the ultrasound beam* is observable -- this
single-component nature is the whole point of a Doppler PINN, and the data loss
must penalise only that component.

Pipeline
--------
1. Beam-direction field: for each sample point, the unit vector from the
   transducer apex to the point (apical view). Multiple windows are stacked.
2. Projection: ``v_beam = dot(u_true, beam_dir)`` -- scalar per point.
3. Sparsity mask: randomly drop a fraction of points (acquisition gaps).
4. Aliasing: wrap ``v_beam`` into ``(-v_nyquist, v_nyquist]`` when it exceeds the
   Nyquist limit; ``dealias=True`` unwraps it again (toggle).
5. Gaussian noise: additive noise at the configured SNR (dB).

Shapes (2D; 3D uses 3-vectors for coords/beam/velocity)
-------------------------------------------------------
Inputs
    coords_true    : (N, 3)   columns (x, y, t)
    velocity_true  : (N, 2)   columns (u, v)
    transducers    : (K, 2)   apex position(s) of each acoustic window
Outputs (:class:`DopplerMeasurements`)
    coords_meas    : (M, 3)          kept sample coordinates
    beam_dir       : (M, 2)          unit beam direction at each kept point
    v_beam         : (M, 1)          measured beam-projected speed [m/s]
    window_id      : (M, 1) long     which transducer produced each sample
    v_beam_clean   : (M, 1)          noise/alias-free projection (diagnostics)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DopplerMeasurements:
    """Sparse single-component Doppler measurements (see module docstring)."""

    coords_meas: torch.Tensor
    beam_dir: torch.Tensor
    v_beam: torch.Tensor
    window_id: torch.Tensor
    v_beam_clean: torch.Tensor


def beam_directions(coords: torch.Tensor, transducer: torch.Tensor) -> torch.Tensor:
    """Unit vectors from ``transducer`` to each point (spatial dims of coords)."""
    dim = transducer.shape[-1]
    d = coords[:, :dim] - transducer
    return d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)


def wrap_to_nyquist(v: torch.Tensor, v_nyquist: float) -> torch.Tensor:
    """Wrap velocities into ``(-v_nyquist, v_nyquist]`` (aliasing)."""
    a = float(v_nyquist)
    return (v + a) % (2.0 * a) - a


def _noise_std_for_snr(clean: torch.Tensor, snr_db: float) -> float:
    """Std of additive Gaussian noise matching a target SNR (dB)."""
    signal_power = float(clean.pow(2).mean().clamp_min(1e-12))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    return noise_power ** 0.5


def synthesize_doppler(
    coords_true: torch.Tensor,
    velocity_true: torch.Tensor,
    transducers: torch.Tensor,
    v_nyquist: float = 0.8,
    snr_db: float = 20.0,
    sparsity: float = 0.5,
    dealias: bool = False,
    rng: Optional[torch.Generator] = None,
) -> DopplerMeasurements:
    """Synthesise sparse, single-component, noisy Doppler measurements.

    See the module docstring for the pipeline and tensor shapes. ``transducers``
    is ``(K, dim)``; each point is measured by every window (stacked), then the
    sparsity mask, aliasing and noise are applied to the pooled set.
    """
    if transducers.dim() == 1:
        transducers = transducers.unsqueeze(0)
    dim = transducers.shape[1]
    dtype = coords_true.dtype

    coords_list, beam_list, vclean_list, win_list = [], [], [], []
    for k in range(transducers.shape[0]):
        beam = beam_directions(coords_true, transducers[k])
        v_beam = (velocity_true[:, :dim] * beam).sum(dim=1, keepdim=True)
        coords_list.append(coords_true)
        beam_list.append(beam)
        vclean_list.append(v_beam)
        win_list.append(torch.full((coords_true.shape[0], 1), k, dtype=torch.long))

    coords = torch.cat(coords_list, dim=0)
    beam = torch.cat(beam_list, dim=0)
    v_clean = torch.cat(vclean_list, dim=0)
    window_id = torch.cat(win_list, dim=0)

    # --- Sparsity mask (acquisition gaps) ---
    if sparsity > 0.0:
        u = torch.rand(coords.shape[0], generator=rng, dtype=dtype)
        keep = u >= sparsity
        if keep.sum() == 0:  # guard against dropping everything
            keep[0] = True
        coords, beam = coords[keep], beam[keep]
        v_clean, window_id = v_clean[keep], window_id[keep]

    # --- Aliasing wraparound (optionally corrected) ---
    v_alias = wrap_to_nyquist(v_clean, v_nyquist)
    v_signal = v_clean if dealias else v_alias

    # --- Gaussian measurement noise at target SNR ---
    std = _noise_std_for_snr(v_clean, snr_db)
    noise = torch.randn(v_signal.shape, generator=rng, dtype=dtype) * std
    v_beam = v_signal + noise

    return DopplerMeasurements(
        coords_meas=coords,
        beam_dir=beam,
        v_beam=v_beam,
        window_id=window_id,
        v_beam_clean=v_clean,
    )
