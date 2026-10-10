"""Low-dimensional activation-driven band forcing for the deliverable Model B.

The *oracle* FSI backbone is handed the true IB forcing ``f`` (evaluation-only
ground truth, unavailable from clinical echo). The **deliverable** Model B may not
use ``f``; its only extra inputs over the kinematic baseline are the accurate FSI
wall velocity (a Dirichlet BC) and a handful of **low-dimensional activation
parameters** (peak active tension ``T_max`` and the contraction *timing* ``g(t)``).

This module turns those into a body-forcing field with a *geometric* spatial
template (so no trajectory information leaks in):

    f(x, t) = T_max * g(t) * w(x) * d_hat(x)

* ``T_max``   -- a single learnable amplitude (softplus, >= 0),
* ``g(t)``    -- a learnable low-dim temporal activation: ``n_harmonics`` sin/cos
  terms of the cardiac period (the "timing"; <= 2*n_harmonics coefficients),
* ``w(x)``    -- a geometric band weight (0 in the cavity interior, ramping to 1 at
  the wall), and
* ``d_hat(x)``-- a geometric inward unit direction,

both supplied by :func:`pinnecho.data.ibfe_dataset.geometric_band_template`, which
is derivable from wall kinematics / cavity geometry alone. The whole forcing is
thus parameterised by ``1 + 2*n_harmonics`` free numbers (~5), co-optimised with
the network, and **never sees the true ``f``**.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActivationForcing(nn.Module):
    """Parametric activation forcing ``f = T_max * g(t) * w(x) * d_hat(x)``."""

    def __init__(self, period: float, n_harmonics: int = 2, init_amp: float = 1.0):
        super().__init__()
        self.period = float(period)
        self.n_harmonics = int(n_harmonics)
        # softplus(raw_amp) == init_amp. Inverse softplus computed stably as
        # raw = init_amp + log(1 - exp(-init_amp)) (avoids overflow of exp(init_amp)
        # for the physical body-force-density scale, which is O(1e4)).
        init_amp = max(float(init_amp), 1e-6)
        raw = init_amp + math.log(-math.expm1(-init_amp))
        self.raw_amp = nn.Parameter(torch.tensor(raw))
        # Temporal harmonic coefficients [a_1, b_1, a_2, b_2, ...].
        coeffs = torch.zeros(2 * self.n_harmonics)
        coeffs[0] = 1.0  # start with a non-trivial first-harmonic sine
        self.coeffs = nn.Parameter(coeffs)

    def amplitude(self) -> torch.Tensor:
        return F.softplus(self.raw_amp)

    def g(self, t: torch.Tensor) -> torch.Tensor:
        """Temporal activation ``g(t)`` (shape ``(N, 1)``), can be positive/negative."""
        w = 2.0 * math.pi / self.period
        out = torch.zeros_like(t)
        for k in range(1, self.n_harmonics + 1):
            a = self.coeffs[2 * (k - 1)]
            b = self.coeffs[2 * (k - 1) + 1]
            out = out + a * torch.sin(k * w * t) + b * torch.cos(k * w * t)
        return out

    def forward(self, t: torch.Tensor, direction: torch.Tensor,
                weight: torch.Tensor) -> torch.Tensor:
        """Body-forcing field ``(N, dim)`` at times ``t`` with geometric template."""
        mag = self.amplitude() * self.g(t) * weight     # (N, 1)
        return mag * direction                           # (N, dim)
