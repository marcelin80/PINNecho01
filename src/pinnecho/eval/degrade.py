"""Contact-free Doppler degradation ladder (our synthesizer, not DUPLEX).

The public DUPLEX ``.mat`` files have no CFD ``(u,w)``. Robustness to
clinical-like Doppler is therefore tested, if at all, by applying
:func:`pinnecho.data.synthesize_doppler.synthesize_doppler` to a
**measured** 3-component volume (Corrado). That is less realistic than
SIMUS; say so.

Do **not** rerun Corrado A/B/C. This module only locks the ladder and
the wrap-fraction gate. A later train, if any, records ``ideal`` first
and reports deltas.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np


# Paper / DUPLEX probe: 2.5 MHz, PRF 7000 Hz, c = 1540 m/s.
DUPLEX_NYQUIST_MPS = 1540.0 * 7000.0 / (4.0 * 2.5e6)

# Locked Corrado GT (STASIS_GT.md). Do not remeasure to move the bar.
CORRADO_SPEED_P99_MPS = 0.261
CORRADO_SPEED_MEAN_MPS = 0.062

# Below this jet-sample wrap fraction the clinical-Nyquist alias step
# is inactive. Corrado p99 0.261 < 1.078, so expect inactive.
WRAP_FRAC_ACTIVE = 0.01

# Artificial Nyquist, labelled ours. Only use if clinical wrap is inactive
# and the text says the wrap is not a bedside PRF.
ALIAS_OURS_MPS = 0.15

LADDER: Dict[str, Dict[str, float]] = {
    "ideal": {
        "v_nyquist": 10.0, "snr_db": 40.0, "sparsity": 0.0, "dealias": True,
    },
    "snr10": {
        "v_nyquist": 10.0, "snr_db": 10.0, "sparsity": 0.0, "dealias": True,
    },
    "sparse50": {
        "v_nyquist": 10.0, "snr_db": 40.0, "sparsity": 0.50, "dealias": True,
    },
    "alias_clinical": {
        "v_nyquist": DUPLEX_NYQUIST_MPS, "snr_db": 40.0,
        "sparsity": 0.0, "dealias": False,
    },
    "alias_ours_15cms": {
        "v_nyquist": ALIAS_OURS_MPS, "snr_db": 40.0,
        "sparsity": 0.0, "dealias": False,
    },
}


def wrap_fraction(v_beam: np.ndarray, v_nyquist: float) -> float:
    """Fraction of samples with ``|v| > v_nyquist`` (would wrap)."""
    v = np.asarray(v_beam, dtype=np.float64).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    return float(np.mean(np.abs(v) > float(v_nyquist)))


def alias_step_is_active(v_beam: np.ndarray, v_nyquist: float) -> bool:
    return wrap_fraction(v_beam, v_nyquist) >= WRAP_FRAC_ACTIVE


def measurement_delta(v_clean: np.ndarray, v_meas: np.ndarray) -> Dict[str, float]:
    """How far the synthesizer moved the *input*, not a PINN score."""
    a = np.asarray(v_clean, dtype=np.float64).reshape(-1)
    b = np.asarray(v_meas, dtype=np.float64).reshape(-1)
    sel = np.isfinite(a) & np.isfinite(b)
    a, b = a[sel], b[sel]
    if a.size == 0:
        return {"n": 0.0, "rmse": float("nan"), "rel_rms": float("nan")}
    err = b - a
    rms_a = float(np.sqrt(np.mean(a * a)))
    return {
        "n": float(a.size),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "rel_rms": float(np.sqrt(np.mean(err * err)) / max(rms_a, 1e-12)),
    }


def apply_wrap(v_clean: np.ndarray, spec: Mapping[str, float]) -> np.ndarray:
    """Nyquist wrap only. For the alias-active gate."""
    v = np.asarray(v_clean, dtype=np.float64)
    if spec.get("dealias", False):
        return v
    a = float(spec["v_nyquist"])
    return (v + a) % (2.0 * a) - a
