"""Public 4D-flow / phantom volume → IBFEFrames (f = 0, not an oracle)."""

import math

import numpy as np
import pytest
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
from pinnecho.data.ibfe_dataset import (
    _stratified_randint, build_doppler_from_frames,
    default_windows_from_frames, make_ibfe_batch_builder,
    per_frame_vbeam_fast_mask, train_ibfe,
)
from pinnecho.data.public_volume import (
    frames_from_public_volume, rasterize_synthetic_lv_3d,
)
from pinnecho.train.public_volume_ab import (
    run_public_volume_ab, run_public_volume_observability,
)

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


def test_public_volume_observability_sweep_smoke():
    frames = synthetic_public_volume_frames(
        Config(), nx=10, ny=10, nz=12, n_frames=2, n_fluid=500, n_wall=120)
    res = run_public_volume_observability(
        frames, windows=(1, 2), steps=8, lbfgs_iters=1, seeds=(0,),
        verbose=False)
    assert res["which"] == "observability"
    assert [r["n_windows"] for r in res["records"]] == [1, 2]
    for r in res["records"]:
        assert math.isfinite(r["vel_relL2_u_mean"])
        assert math.isnan(r["pressure_relL2_mean"])


def test_stratified_randint_respects_fast_frac():
    mask = torch.zeros(1000, dtype=torch.bool)
    mask[:200] = True
    g = torch.Generator().manual_seed(0)
    idx = _stratified_randint(200, mask, g, 0.5)
    assert idx.numel() == 200
    assert float(mask[idx].float().mean()) == pytest.approx(0.5, abs=1e-6)


def test_data_fast_frac_oversamples_abs_v_beam():
    frames = synthetic_public_volume_frames(
        Config(), nx=10, ny=10, nz=12, n_frames=2, n_fluid=400, n_wall=80)
    transducers = default_windows_from_frames(frames, n_windows=1)
    _, _, vb = build_doppler_from_frames(
        frames, transducers, noise_level=0.0, seed=0)
    thresh = float(vb.abs().quantile(0.8))
    pool = float((vb.abs() >= thresh).float().mean())
    build = make_ibfe_batch_builder(
        frames, transducers=transducers, n_data=80, noise_level=0.0,
        data_fast_frac=0.5, data_fast_thresh=thresh, seed=0)
    hits = [float((build(i)["data"]["v_beam"].abs() >= thresh).float().mean())
            for i in range(15)]
    assert float(np.mean(hits)) > pool
    assert float(np.mean(hits)) == pytest.approx(0.5, abs=0.12)


def test_per_frame_vbeam_fast_mask_is_a_rank_not_a_cm_cut():
    # Two frames, 100 rows each. Frame 0 is slow (max 4 cm/s); frame 1 is
    # fast (max 40 cm/s). A 10 cm/s cut would mark frame 0 empty and most
    # of frame 1. The per-frame quantile marks ~20 % of *each* frame.
    vb = torch.zeros(200)
    t = torch.cat([torch.zeros(100), torch.ones(100)])
    vb[:100] = torch.linspace(0.0, 0.04, 100)
    vb[100:] = torch.linspace(0.0, 0.40, 100)
    mask = per_frame_vbeam_fast_mask(vb, t, 0.80)
    assert int(mask[:100].sum()) == pytest.approx(20, abs=1)
    assert int(mask[100:].sum()) == pytest.approx(20, abs=1)
    # Independent of 5/10/15 cm/s: the slow frame still contributes.
    assert not bool((vb[:100].abs() >= 0.10).any())
    assert bool(mask[:100].any())


def test_data_fast_frac_uses_per_frame_quantile_when_set():
    frames = synthetic_public_volume_frames(
        Config(), nx=10, ny=10, nz=12, n_frames=2, n_fluid=400, n_wall=80)
    transducers = default_windows_from_frames(frames, n_windows=1)
    X, _, vb = build_doppler_from_frames(
        frames, transducers, noise_level=0.0, seed=0)
    dim = frames.spatial_dim
    times = X[:, dim]
    fast = per_frame_vbeam_fast_mask(vb, times, 0.80)
    pool = float(fast.float().mean())
    cuts = {}
    for tv in torch.unique(times):
        cuts[float(tv)] = float(torch.quantile(vb.reshape(-1)[times == tv].abs(), 0.80))
    build = make_ibfe_batch_builder(
        frames, transducers=transducers, n_data=80, noise_level=0.0,
        data_fast_frac=0.5, data_fast_quantile=0.80, seed=0)
    hits = []
    for i in range(15):
        batch = build(i)
        vb_b = batch["data"]["v_beam"].abs().reshape(-1)
        t_b = batch["data"]["X"][:, dim]
        marked = torch.zeros_like(vb_b, dtype=torch.bool)
        for tv, cut in cuts.items():
            marked[t_b == tv] = vb_b[t_b == tv] >= cut
        hits.append(float(marked.float().mean()))
    assert pool < 0.35
    assert float(np.mean(hits)) > pool
    assert float(np.mean(hits)) == pytest.approx(0.5, abs=0.15)


def test_oracle_gt_speed_cap_still_works():
    t_len, n = 2, 10
    vel = np.zeros((t_len, n, n, n, 3))
    mask = np.ones((n, n, n), dtype=bool)
    vel[:, :2, :, :, 0] = 0.40  # 20 % of the cube is fast
    kwargs = dict(
        spacing=(0.002, 0.002, 0.002), origin=(0, 0, 0),
        times=(0.0, 0.3), n_fluid=200, n_wall=40, seed=0)
    uniform = frames_from_public_volume(vel, mask, **kwargs)
    strat = frames_from_public_volume(
        vel, mask, fast_frac=0.5, fast_thresh=0.10, **kwargs)
    u = float((uniform.velocity_fluid.norm(dim=1) >= 0.10).float().mean())
    s = float((strat.velocity_fluid.norm(dim=1) >= 0.10).float().mean())
    assert u == pytest.approx(0.20, abs=0.08)
    assert s == pytest.approx(0.50, abs=0.08)
    assert s > u


def test_fast_frac_none_matches_uniform_cap():
    t_len, n = 2, 8
    vel = np.zeros((t_len, n, n, n, 3))
    mask = np.ones((n, n, n), dtype=bool)
    vel[:, :2, :, :, 0] = 0.40
    a = frames_from_public_volume(
        vel, mask, spacing=(0.002, 0.002, 0.002), origin=(0, 0, 0),
        times=(0.0, 0.3), n_fluid=80, n_wall=20, seed=1)
    b = frames_from_public_volume(
        vel, mask, spacing=(0.002, 0.002, 0.002), origin=(0, 0, 0),
        times=(0.0, 0.3), n_fluid=80, n_wall=20, seed=1, fast_frac=None)
    assert torch.equal(a.coords_fluid, b.coords_fluid)


def test_empty_mask_raises():
    import pytest
    vel = np.zeros((2, 6, 6, 6, 3))
    mask = np.zeros((2, 6, 6, 6), dtype=bool)
    with pytest.raises(ValueError, match="no interior"):
        frames_from_public_volume(
            vel, mask, spacing=(0.01, 0.01, 0.01), origin=(0, 0, 0),
            times=(0.0, 0.3))
