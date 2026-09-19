"""TODO STUB: synthesise Doppler-like measurements from a ground-truth field.

Turns a dense ground-truth velocity field (from IBFE output, or the Stage-1
synthetic LV) into the sparse, single-component, noisy measurements a colour /
spectral Doppler acquisition would produce from an apical window. Only the
component of velocity *along the ultrasound beam* is observable -- this
single-component nature is the whole point of a Doppler PINN, and the data loss
must penalise only that component.

This is left as a clearly-marked stub because the exact acquisition geometry
(number of transducers/windows, beam layout, Nyquist limit, SNR, sparsity) will
be finalised with the real IBFE data. The Stage-1 working implementation lives
in :mod:`pinnecho.data.doppler`; this module documents the canonical shapes and
the operations the real synthesiser must perform.

Pipeline (each step is a documented TODO)
-----------------------------------------
1. Beam-direction field: for each sample point, unit vector from the transducer
   apex to the point (apical view). Multiple windows/transducers may be stacked.
2. Projection: v_beam = dot(u_true, beam_dir)  -- scalar per point.
3. Sparsity mask: randomly drop a fraction of points (acquisition gaps / dropout).
4. Aliasing: wrap v_beam into (-v_nyquist, v_nyquist] when |v_beam| exceeds the
   Nyquist limit; ``dealias=True`` optionally unwraps it again (toggle).
5. Gaussian noise: add noise at the configured SNR (default ~20 dB).

Expected shapes (2D; 3D uses 3-vectors for coords/beam)
-------------------------------------------------------
Inputs
    coords_true    : (N, 3)   columns (x, y, t)
    velocity_true  : (N, 2)   columns (u, v)
    transducers    : (K, 2)   apex position(s) of each acoustic window
    v_nyquist      : float    aliasing limit [m/s]
    snr_db         : float    measurement SNR in dB
    sparsity       : float    fraction of points dropped in [0, 1)
    dealias        : bool     whether to unwrap aliased velocities
    rng            : torch.Generator

Outputs (a :class:`DopplerMeasurements` bundle)
    coords_meas    : (M, 3)          kept sample coordinates
    beam_dir       : (M, 2)          unit beam direction at each kept point
    v_beam         : (M, 1)          measured beam-projected speed [m/s]
    window_id      : (M, 1) long     which transducer produced each sample
    v_beam_clean   : (M, 1)          noise/alias-free projection (for diagnostics)
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

    See the module docstring for the exact pipeline and tensor shapes.

    Notes
    -----
    TODO: implement the finalised acquisition model here (multi-window beam
    geometry, aliasing wraparound + optional dealiasing, sparsity mask, SNR
    noise). The Stage-1 stand-in with a working single/dual-window model is
    :class:`pinnecho.data.doppler.DopplerSampler`; consolidate into this
    function once the real acquisition parameters are fixed.
    """
    raise NotImplementedError(
        "synthesize_doppler is a documented stub. See pinnecho.data.doppler "
        "for the working Stage-1 acquisition model, and this module's docstring "
        "for the canonical shapes/pipeline to implement against real IBFE data."
    )
