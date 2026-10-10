"""Virtual Doppler acquisition: sparse, single-component velocity samples.

A real color/PW-Doppler system only measures the velocity component *along the
ultrasound beam* (the radial velocity relative to the transducer). We emulate
this by projecting the true velocity onto the unit vector pointing from a
virtual transducer to each sample point, then adding measurement noise and
keeping only a sparse set of samples per time frame. This one-component,
sparse, noisy signal is the *only* velocity information the PINN sees.
"""

import math
from typing import Dict

import numpy as np
import torch

from ..config import DopplerConfig
from .synthetic_lv import SyntheticLVFSI


def project_velocity(u: torch.Tensor, v: torch.Tensor, beam: torch.Tensor) -> torch.Tensor:
    """Project velocity ``(u, v)`` onto unit beam directions ``beam`` (``(N, 2)``)."""
    return u * beam[:, 0:1] + v * beam[:, 1:2]


class DopplerSampler:
    """Draws sparse single-component Doppler samples from an LV FSI field."""

    def __init__(self, cfg: DopplerConfig):
        self.cfg = cfg

    def beam_directions(self, X: torch.Tensor, transducer=None) -> torch.Tensor:
        """Unit vectors from a virtual transducer to each sample point."""
        transducer = transducer if transducer is not None else self.cfg.transducer
        tx = torch.tensor([transducer[0], transducer[1]], dtype=X.dtype)
        d = X[:, 0:2] - tx
        d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return d

    def sample(self, lv: SyntheticLVFSI, rng: np.random.Generator) -> Dict[str, torch.Tensor]:
        """Return a dictionary describing the (possibly multi-window) acquisition.

        For each acoustic window a fresh set of interior points is drawn per
        frame and the velocity is projected onto that window's beam directions.
        Measurements from all windows are concatenated.
        """
        period = lv.flow.period
        t_frames = np.linspace(
            self.cfg.t_start_frac * period,
            self.cfg.t_end_frac * period,
            self.cfg.n_frames,
        )
        windows = self.cfg.all_windows()
        X_list, beam_list = [], []
        for transducer in windows:
            coords = []
            for t in t_frames:
                coords.append(
                    lv.sample_interior(self.cfg.n_points_per_frame, np.array([t]), rng)
                )
            Xw = torch.cat(coords, dim=0)
            X_list.append(Xw)
            beam_list.append(self.beam_directions(Xw, transducer))
        X = torch.cat(X_list, dim=0)
        beam = torch.cat(beam_list, dim=0)

        fields = lv.all_fields(X)
        doppler_true = project_velocity(fields["u"], fields["v"], beam)

        # Additive Gaussian noise scaled by the RMS Doppler magnitude.
        rms = float(torch.sqrt((doppler_true ** 2).mean()).clamp_min(1e-9))
        noise = torch.as_tensor(
            rng.normal(0.0, self.cfg.noise_level * rms, size=doppler_true.shape),
            dtype=X.dtype,
        )
        doppler = doppler_true + noise

        return {
            "X": X,
            "beam": beam,
            "doppler": doppler,
            "doppler_true": doppler_true,
            "u_true": fields["u"],
            "v_true": fields["v"],
            "t_frames": torch.as_tensor(t_frames, dtype=X.dtype),
            "noise_rms": torch.tensor(self.cfg.noise_level * rms, dtype=X.dtype),
        }
