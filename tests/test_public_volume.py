"""Public 4D-flow / phantom volume → IBFEFrames (f = 0, not an oracle)."""

import math

import numpy as np
import torch

from pinnecho.config import Config
from pinnecho.data import (
    has_oracle_forcing,
    is_public_volume_npz,
    load_ibfe_output,
    save_public_volume_npz,
    synthetic_public_volume_frames,
    validate_ibfe_frames,
)
from pinnecho.data.ibfe_dataset import train_ibfe
from pinnecho.data.public_volume import (
    frames_from_public_volume, rasterize_synthetic_lv_3d,
)
from pinnecho.train.public_volume_ab import run_public_volume_ab

torch.set_default_dtype(torch.float32)


def test_public_volume_has_zero_forcing_and_traction():
    frames = synthetic_public_volume_frames(
        Config(), nx=12, ny=12, nz=14, n_frames=3, n_fluid=800, n_wall=200)
    assert frames.spatial_dim == 3
    assert not has_oracle_forcing(frames)
    assert float(frames.forcing_fluid.abs().max()) == 0.0
    assert float(frames.traction_wall.abs().max()) == 0.0
    assert frames.coords_fluid.shape[0] > 0
    assert frames.coords_wall.shape[0] > 0
    assert frames.velocity_fluid.shape[1] == 3


def test_wall_normals_are_unit_and_outward():
    frames = synthetic_public_volume_frames(
        Config(), nx=14, ny=14, nz=16, n_frames=2, n_fluid=1000, n_wall=300)
    norms = frames.normals_wall.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    # Outward: wall - cavity centroid should have nonnegative mean dot with n.
    centroid = frames.coords_wall[:, :3].mean(dim=0)
    radial = frames.coords_wall[:, :3] - centroid
    dots = (radial * frames.normals_wall).sum(dim=1)
    assert float(dots.mean()) > 0.0


def test_pressure_absent_is_zero_present_is_not():
    cfg = Config()
    no_p = synthetic_public_volume_frames(
        cfg, nx=10, ny=10, nz=12, n_frames=2, n_fluid=400, n_wall=80,
        with_pressure=False)
    assert float(no_p.pressure_fluid.abs().max()) == 0.0
    yes_p = synthetic_public_volume_frames(
        cfg, nx=10, ny=10, nz=12, n_frames=2, n_fluid=400, n_wall=80,
        with_pressure=True)
    # Manufactured p is not identically zero.
    assert float(yes_p.pressure_fluid.abs().max()) > 0.0


def test_public_volume_npz_roundtrip_and_dispatch(tmp_path):
    volume, _ = rasterize_synthetic_lv_3d(
        Config(), nx=10, ny=10, nz=12, n_frames=2, with_pressure=False)
    path = tmp_path / "volume.npz"
    save_public_volume_npz(
        path, velocity=volume["velocity"], mask=volume["mask"],
        spacing=volume["spacing"], origin=volume["origin"], times=volume["times"],
        cycle_period=volume["cycle_period"], rho=volume["rho"], mu=volume["mu"])
    assert is_public_volume_npz(path)
    loaded = load_ibfe_output(str(path), n_fluid=500, n_wall=120)
    assert loaded.spatial_dim == 3
    assert not has_oracle_forcing(loaded)
    report = validate_ibfe_frames(loaded)
    assert report.ok
    assert any("public 4D-flow" in w for w in report.warnings)


def test_rejects_oracle_bundle_on_public_ab_sweep():
    from pinnecho.data import synthetic_ibfe_frames_3d
    import pytest
    frames = synthetic_ibfe_frames_3d(Config(), n_fluid=400, n_wall=80, n_frames=2)
    assert has_oracle_forcing(frames)
    with pytest.raises(ValueError, match="public volumes"):
        run_public_volume_ab(frames, steps=2, lbfgs_iters=1, seeds=(0,),
                             verbose=False)


def test_public_volume_train_smoke_baseline_and_fsi_param():
    frames = synthetic_public_volume_frames(
        Config(), nx=12, ny=12, nz=14, n_frames=3, n_fluid=700, n_wall=180)
    for backbone in ("baseline", "fsi_param"):
        _, metrics = train_ibfe(
            frames, backbone=backbone, steps=8, lbfgs_iters=1,
            use_traction=False, seed=0, verbose=False)
        assert all(math.isfinite(v) for v in metrics.values()
                   if isinstance(v, float))


def test_public_volume_ab_sweep_smoke():
    frames = synthetic_public_volume_frames(
        Config(), nx=10, ny=10, nz=12, n_frames=2, n_fluid=500, n_wall=120)
    res = run_public_volume_ab(
        frames, steps=8, lbfgs_iters=1, seeds=(0,), verbose=False)
    assert res["has_oracle_forcing"] is False
    assert res["has_pressure_gt"] is False
    assert [r["backbone"] for r in res["records"]] == ["baseline", "fsi_param"]
    assert "deliverable_vs_baseline" in res["contrasts"]
    for r in res["records"]:
        assert math.isfinite(r["vel_relL2_speed_mean"])
        assert math.isnan(r["pressure_relL2_mean"])


def test_empty_mask_raises():
    import pytest
    vel = np.zeros((2, 6, 6, 6, 3))
    mask = np.zeros((2, 6, 6, 6), dtype=bool)
    with pytest.raises(ValueError, match="no interior"):
        frames_from_public_volume(
            vel, mask, spacing=(0.01, 0.01, 0.01), origin=(0, 0, 0),
            times=(0.0, 0.3))
