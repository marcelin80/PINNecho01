"""Contact-free degradation ladder (no MATLAB, no Corrado NPZ, no train)."""

import math

import numpy as np
import pytest

from pinnecho.eval.degrade import (
    ALIAS_OURS_MPS, CORRADO_SPEED_P99_MPS, DUPLEX_NYQUIST_MPS, LADDER,
    WRAP_FRAC_ACTIVE, alias_step_is_active, measurement_delta, wrap_fraction,
)


def test_duplex_nyquist_matches_probe():
    assert DUPLEX_NYQUIST_MPS == pytest.approx(1.078, abs=5e-4)


def test_corrado_p99_is_below_clinical_nyquist():
    assert CORRADO_SPEED_P99_MPS < DUPLEX_NYQUIST_MPS
    # A field that never exceeds p99 cannot wrap at the DUPLEX rail.
    v = np.full(1000, CORRADO_SPEED_P99_MPS)
    assert wrap_fraction(v, DUPLEX_NYQUIST_MPS) == 0.0
    assert alias_step_is_active(v, DUPLEX_NYQUIST_MPS) is False


def test_wrap_fraction_counts_exceedances():
    v = np.array([0.10, 0.20, 1.50, -1.20])
    assert wrap_fraction(v, 1.0) == pytest.approx(0.50)


def test_ours_nyquist_is_labelled_and_below_p99():
    assert ALIAS_OURS_MPS < CORRADO_SPEED_P99_MPS
    assert LADDER["alias_ours_15cms"]["v_nyquist"] == ALIAS_OURS_MPS


def test_ladder_keys_are_locked():
    assert set(LADDER) == {
        "ideal", "snr10", "sparse50", "alias_clinical", "alias_ours_15cms",
    }
    assert math.isclose(WRAP_FRAC_ACTIVE, 0.01)


def test_measurement_delta_zero_on_copy():
    v = np.linspace(-0.2, 0.2, 50)
    d = measurement_delta(v, v.copy())
    assert d["rmse"] == pytest.approx(0.0)
    assert d["n"] == 50
