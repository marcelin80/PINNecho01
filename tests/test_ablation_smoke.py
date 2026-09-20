"""Smoke tests for the FSI-backbone control ablations.

These run a few training steps only; they check that the drivers execute
end-to-end and return the expected structure, not the scientific magnitudes.
"""

import torch

from pinnecho.config import Config
from pinnecho.train.ablation import (
    perturb_traction,
    run_physics_isolation,
    run_traction_perturbation,
    run_pressure_observability,
    run_observability_physics,
    forcing_pressure_coupling,
    _apply_windows,
    _pearson,
    _aggregate,
    _fmt_table,
)


def test_perturb_traction_levels():
    torch.manual_seed(0)
    t = torch.randn(200, 2)
    # eps=0 returns the exact target unchanged.
    assert torch.equal(perturb_traction(t, 0.0), t)
    # Perturbation magnitude grows with the level.
    d05 = (perturb_traction(t, 0.05, seed=0) - t).norm()
    d20 = (perturb_traction(t, 0.20, seed=0) - t).norm()
    assert 0.0 < d05 < d20


def test_aggregate_mean_std():
    agg = _aggregate([{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}])
    assert abs(agg["a"][0] - 2.0) < 1e-9
    assert agg["a"][1] > 0.0  # std of {1,3} is 1.0


def test_physics_isolation_tiny():
    torch.set_default_dtype(torch.float32)
    cfg = Config()
    res = run_physics_isolation(cfg, seeds=(0,), steps=30, lbfgs_iters=0,
                                verbose=False)
    for name in ("A_noisy", "A_exact", "B_forcing", "B_forcing_traction"):
        assert name in res
        assert "pressure_corr" in res[name]
        mean, std = res[name]["pressure_corr"]
        assert -1.0 <= mean <= 1.0 and std >= 0.0
    # Table renders without error.
    assert "variant" in _fmt_table(res)


def test_traction_perturbation_tiny():
    torch.set_default_dtype(torch.float32)
    cfg = Config()
    res = run_traction_perturbation(cfg, seeds=(0,), levels=(0.0, 0.2),
                                    steps=30, lbfgs_iters=0, verbose=False)
    assert "eps=0.00" in res and "eps=0.20" in res
    assert "pressure_relL2" in res["eps=0.20"]


def test_pearson_and_windows():
    assert abs(_pearson([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-9
    assert abs(_pearson([1, 2, 3], [6, 4, 2]) + 1.0) < 1e-9
    cfg = Config()
    assert _apply_windows(cfg, 1).doppler.all_windows().__len__() == 1
    assert len(_apply_windows(cfg, 3).doppler.all_windows()) == 3


def test_forcing_pressure_coupling():
    torch.set_default_dtype(torch.float64)
    cfg = Config()
    c = forcing_pressure_coupling(cfg, seed=0, n=800)
    # The manufactured forcing contains grad p explicitly, so it is strongly
    # aligned with the pressure gradient (circularity is real and measurable).
    assert c["corr_forcing_gradp"] > c["corr_inertialviscous_gradp"]
    assert 0.0 <= c["gradp_magnitude_fraction"] <= 5.0


def test_pressure_observability_tiny():
    torch.set_default_dtype(torch.float32)
    cfg = Config()
    per_window, coupling = run_pressure_observability(
        cfg, seeds=(0,), windows=(1, 2), steps=25, lbfgs_iters=0, verbose=False)
    assert "windows=1" in per_window and "windows=2" in per_window
    assert "corr_velU_pressureCorr" in coupling


def test_substitution_tiny():
    torch.set_default_dtype(torch.float32)
    cfg = Config()
    res = run_observability_physics(cfg, seeds=(0,), windows=(1, 2),
                                    steps=25, lbfgs_iters=0, verbose=False)
    for k in ("w1_baseline", "w1_forcing", "w2_baseline", "w2_forcing"):
        assert k in res
