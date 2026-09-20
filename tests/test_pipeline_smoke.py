"""End-to-end smoke tests for the model, trainer and evaluation pipeline."""

import numpy as np
import torch

from pinnecho.config import Config
from pinnecho.data.dataset import build_dataset
from pinnecho.models.pinn import PINN
from pinnecho.pipeline import build_lv, train_model, evaluate_model


def _tiny_cfg(backbone="baseline") -> Config:
    c = Config()
    c.doppler.n_points_per_frame = 40
    c.doppler.n_frames = 4
    c.collocation.n_interior = 400
    c.collocation.n_wall = 120
    c.collocation.n_frames = 6
    c.model.hidden_width = 32
    c.model.hidden_depth = 2
    c.model.fourier_features = 8
    c.physics.backbone = backbone
    c.train.iterations = 60
    c.train.log_every = 60
    c.train.batch_interior = 256
    c.train.batch_wall = 64
    return c


def test_model_forward_shapes():
    c = _tiny_cfg()
    ds = build_dataset(c)
    model = PINN(c.model, ds.scales).to(torch.float64)
    u, v, p = model(ds.meas_X)
    assert u.shape == v.shape == p.shape == (ds.meas_X.shape[0], 1)


def test_backbone_switch_changes_forcing():
    c_base = _tiny_cfg("baseline")
    c_fsi = _tiny_cfg("fsi_informed")
    ds = build_dataset(c_base)
    assert ds.col_forcing.abs().max() > 0
    # Baseline ignores forcing; FSI uses it -> trainers differ.
    from pinnecho.train.trainer import Trainer

    t_base = Trainer(c_base, build_dataset(c_base))
    t_fsi = Trainer(c_fsi, build_dataset(c_fsi))
    assert t_base.forcing is None
    assert t_fsi.forcing is not None


def test_training_reduces_loss():
    c = _tiny_cfg()
    ds = build_dataset(c)
    model, state = train_model(c, ds)
    totals = state.history["total"]
    assert totals[-1] < totals[0]


def test_evaluation_returns_expected_keys():
    c = _tiny_cfg()
    lv = build_lv(c)
    ds = build_dataset(c, lv=lv)
    model, _ = train_model(c, ds)
    metrics = evaluate_model(model, lv, c, with_residence_time=True)
    for key in [
        "rel_l2_speed", "rel_l2_pressure", "rel_l2_vorticity", "rel_l2_wss",
        "rel_l2_residence_time",
    ]:
        assert key in metrics
        assert np.isfinite(metrics[key])


def test_dataset_save_load_roundtrip(tmp_path):
    from pinnecho.data.dataset import ReconstructionDataset

    c = _tiny_cfg()
    ds = build_dataset(c)
    path = tmp_path / "ds.npz"
    ds.save(path)
    ds2 = ReconstructionDataset.load(path)
    assert torch.allclose(ds.meas_doppler, ds2.meas_doppler)
    assert torch.allclose(ds.col_forcing, ds2.col_forcing)
    assert abs(ds.scales.velocity - ds2.scales.velocity) < 1e-12
