"""Tests for the deliverable Model B (fsi_param): low-dim activation forcing.

The deliverable Model B uses only wall kinematics + a low-dim activation ansatz
(T_max + timing g(t)) with a geometric band template -- it never sees the true
forcing f (that is the oracle). These cover the ActivationForcing module, the
geometric template, the backbone wiring, and the 3-way deliverable sweep.
"""

import math

import torch

from pinnecho.config import Config
from pinnecho.data import synthetic_ibfe_frames_3d, geometric_band_template
from pinnecho.data.ibfe_dataset import build_model_for_ibfe, train_ibfe
from pinnecho.models import ActivationForcing
from pinnecho.train.band_oracle import (
    DELIVERABLE_CONTRASTS,
    deliverable_conditions,
    run_band_oracle_sweep,
)

torch.set_default_dtype(torch.float32)


def test_activation_forcing_is_low_dim_and_shaped():
    af = ActivationForcing(period=0.9, n_harmonics=2, init_amp=1000.0)
    # 1 amplitude + 2*n_harmonics coefficients == 5 free parameters.
    assert sum(p.numel() for p in af.parameters()) == 5
    assert float(af.amplitude().detach()) >= 0.0
    # inverse-softplus init recovers the requested amplitude (no overflow at 1e3+).
    assert abs(float(af.amplitude().detach()) - 1000.0) < 1.0
    t = torch.linspace(0, 0.9, 20).reshape(-1, 1)
    d = torch.randn(20, 3)
    d = d / d.norm(dim=1, keepdim=True)
    w = torch.rand(20, 1)
    f = af(t, d, w)
    assert f.shape == (20, 3)


def test_geometric_template_is_inward_unit_and_banded():
    cfg = Config()
    frames = synthetic_ibfe_frames_3d(cfg, n_fluid=1500, n_wall=300, n_frames=4)
    d, w = geometric_band_template(frames.coords_fluid, 3, band_frac=0.35)
    norms = d.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert float(w.min()) >= 0.0 and float(w.max()) <= 1.0
    # Some interior points have zero weight; some near-wall points reach ~1.
    assert float(w.min()) == 0.0 and float(w.max()) > 0.5


def test_fsi_param_backbone_wires_activation_forcing():
    cfg = Config()
    frames = synthetic_ibfe_frames_3d(cfg, n_fluid=800, n_wall=200, n_frames=3)
    model, loss_fn = build_model_for_ibfe(frames, backbone="fsi_param")
    assert isinstance(model.activation_forcing, ActivationForcing)
    assert loss_fn.forcing == "param"
    assert loss_fn.wall_mode == "fsi"               # accurate FSI wall BC is allowed
    # The activation params are part of model.parameters() (so the optimiser trains
    # them jointly with the network).
    model_param_ids = {id(p) for p in model.parameters()}
    assert all(id(p) in model_param_ids for p in model.activation_forcing.parameters())
    assert all(p.requires_grad for p in model.activation_forcing.parameters())


def test_fsi_param_training_smoke_runs():
    cfg = Config()
    frames = synthetic_ibfe_frames_3d(cfg, n_fluid=1000, n_wall=250, n_frames=3,
                                      forcing_band_frac=0.2)
    model, metrics = train_ibfe(frames, backbone="fsi_param", steps=12, lbfgs_iters=1,
                                use_traction=False, seed=0, verbose=False)
    assert all(metrics[k] == metrics[k] for k in metrics)      # no NaNs
    assert float(model.activation_forcing.amplitude().detach()) >= 0.0


def test_deliverable_conditions_and_sweep():
    conds = deliverable_conditions()
    backbones = [c.backbone for c in conds]
    assert backbones == ["baseline", "fsi_param", "fsi_informed"]
    res = run_band_oracle_sweep(
        Config(), conditions=conds, contrasts=DELIVERABLE_CONTRASTS,
        dim=2, band_frac=0.25, steps=10, lbfgs_iters=1, seeds=(0,),
        use_traction=False, n_fluid=800, n_wall=200, n_frames=3, n_windows=2,
        verbose=False)
    assert len(res["records"]) == 3
    assert set(res["contrasts"]) == set(DELIVERABLE_CONTRASTS)
    for r in res["records"]:
        assert math.isfinite(r["pressure_relL2_mean"])
