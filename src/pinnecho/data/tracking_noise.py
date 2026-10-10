"""Named wall-tracking error presets for Model A's kinematic wall BC.

In a real study Model A's wall velocity is *not* the FSI structural interface
velocity: it is a contour-tracking estimate (segmented endocardium, finite-
differenced in time) and therefore carries a systematic under-estimation plus
test-retest scatter. Stage 1 used a placeholder ``(-8% bias, 10% RMS noise)``.
This module replaces that magic pair with **named, citable presets** so every
downstream call site can say ``tracking="ste"`` or ``tracking="exact"`` instead
of sprinkling raw numbers.

Presets
-------
``exact``
    No error. This is the ``A_exact`` / coincident-wall isolation condition
    (Ablation 1): both backbones see the true endocardial velocity, so any
    remaining A/B gap is a physics-term effect, not a wall-BC effect.
``placeholder``
    Legacy Stage-1 pair ``bias = -0.08``, ``noise = 0.10``. Kept as the
    **default** so published Ablation 1–6 numbers stay reproducible.
``ste``
    Speckle-tracking echocardiography (STE) literature:

    * **noise = 0.09** — Houard et al. *Clin Res Cardiol* (2021) STE-LV-GLS
      test-retest coefficient of variation **8.9%** (ICC 0.94). Sits at the
      upper end of Farsalinos et al. *JASE* (2015) inter-observer relative
      mean error for three-view GLS (**5.4–8.6%**; intra-observer 4.9–7.3%).
      GLS is a displacement-derived global wall-motion metric, so its
      test-retest CoV is the right scale for a contour-tracking wall-velocity
      perturbation (velocity is a time derivative of the same tracked contour).
    * **bias = -0.05** — modest systematic under-estimation. Amundsen et al.
      *JACC* (2006) reported STE underestimating short-axis shortening at
      higher values vs sonomicrometry (long-axis was unbiased); contour
      finite-differencing + temporal smoothing pulls peak wall *speed* the
      same way. Vendor-dependent STE-vs-tagged-MRI bias is *not* a single
      number (Obert et al. *Circ Cardiovasc Imaging* 2018 saw large
      direction-dependent relative bias), so we take a conservative 5%
      under-estimation rather than claiming a unique truth offset.

The Doppler *measurement* noise (``DopplerConfig.noise_level``, default 5% of
mean |v_beam|) is a separate knob — this module only corrupts the **wall BC**.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass(frozen=True)
class TrackingNoise:
    """Multiplicative bias + additive RMS noise applied to a wall-velocity field.

    Realised as ``v_kin = v_fsi * (1 + bias) + N(0, (noise * rms(v_fsi))^2)``.
    ``bias < 0`` is a systematic under-estimation of peak wall speed.
    """

    name: str
    bias: float
    noise: float
    sources: str = ""


PRESETS = {
    "exact": TrackingNoise(
        name="exact", bias=0.0, noise=0.0,
        sources="no tracking error; A_exact / coincident-wall isolation"),
    "placeholder": TrackingNoise(
        name="placeholder", bias=-0.08, noise=0.10,
        sources="legacy Stage-1 placeholder (8% under-estimation + 10% RMS); "
                "kept so Ablation 1-6 numbers stay reproducible"),
    "ste": TrackingNoise(
        name="ste", bias=-0.05, noise=0.09,
        sources="Houard 2021 STE-LV-GLS test-retest CV 8.9% (noise); "
                "Farsalinos 2015 JASE inter-observer GLS 5.4-8.6%; "
                "Amundsen 2006 JACC short-axis underestimation (bias -5%)"),
}

PresetLike = Union[str, TrackingNoise, None]


def resolve_tracking(preset: PresetLike = None, *,
                     bias: Optional[float] = None,
                     noise: Optional[float] = None) -> TrackingNoise:
    """Resolve a preset name (or ``TrackingNoise``) plus optional overrides."""
    if isinstance(preset, TrackingNoise):
        base = preset
    elif preset is None:
        base = PRESETS["placeholder"]
    elif isinstance(preset, str):
        key = preset.strip().lower()
        if key not in PRESETS:
            raise KeyError(
                f"unknown tracking preset {preset!r}; "
                f"expected one of {sorted(PRESETS)}")
        base = PRESETS[key]
    else:
        raise TypeError(f"preset must be str | TrackingNoise | None, got {type(preset)}")
    return TrackingNoise(
        name=base.name,
        bias=base.bias if bias is None else float(bias),
        noise=base.noise if noise is None else float(noise),
        sources=base.sources,
    )


def apply_tracking_noise(velocity: torch.Tensor, tracking: PresetLike = "ste",
                         *, seed: int = 0,
                         bias: Optional[float] = None,
                         noise: Optional[float] = None) -> torch.Tensor:
    """Return a contour-tracking estimate of ``velocity`` under ``tracking``.

    ``tracking="exact"`` (or a zero-error spec) returns ``velocity`` unchanged
    and does not touch the RNG. The output has the same shape / dtype / device
    as ``velocity`` and does not share storage with it.
    """
    spec = resolve_tracking(tracking, bias=bias, noise=noise)
    if spec.bias == 0.0 and spec.noise == 0.0:
        return velocity.detach().clone()
    rms = float(velocity.pow(2).mean().sqrt().clamp_min(1e-9))
    g = torch.Generator(device="cpu").manual_seed(int(seed) + 7)
    # Sample on CPU in the velocity dtype so placeholder matches the legacy
    # Stage-1 realisation (Ablation 1-6 used float32 randn at seed+7).
    noise_cpu = torch.randn(velocity.shape, generator=g, dtype=velocity.dtype)
    noise_t = noise_cpu.to(device=velocity.device) * (spec.noise * rms)
    return velocity * (1.0 + spec.bias) + noise_t


def relative_rms_error(estimate: torch.Tensor, truth: torch.Tensor) -> float:
    """``||estimate - truth||_rms / ||truth||_rms`` (0 if truth is ~0)."""
    denom = float(truth.pow(2).mean().sqrt())
    if denom < 1e-12:
        return 0.0
    return float((estimate - truth).pow(2).mean().sqrt()) / denom
