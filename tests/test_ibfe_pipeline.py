"""IBFE data-preparation pipeline: I/O, validation, and the training adapter."""

import dataclasses

import numpy as np
import torch

from pinnecho.config import Config
from pinnecho.data.load_ibfe_output import (
    synthetic_ibfe_frames, synthetic_ibfe_frames_3d, load_ibfe_output,
)
from pinnecho.data.ibfe_io import (
    save_ibfe_npz, load_ibfe_npz, save_ibfe_manifest_csv, load_ibfe_manifest,
    NPZ_TENSOR_KEYS,
)
from pinnecho.data.ibfe_validate import validate_ibfe_frames
from pinnecho.data.ibfe_dataset import (
    make_ibfe_batch_builder, build_model_for_ibfe, evaluate_ibfe, train_ibfe,
    default_windows_from_frames, build_doppler_from_frames,
)


def _frames_2d(with_valve=True):
    torch.set_default_dtype(torch.float64)
    cfg = Config()
    return synthetic_ibfe_frames(cfg, n_fluid=600, n_wall=200, n_frames=5,
                                 with_valve=with_valve)


def test_npz_roundtrip(tmp_path):
    frames = _frames_2d()
    path = save_ibfe_npz(frames, tmp_path / "bundle.npz")
    loaded = load_ibfe_npz(path)
    for k in NPZ_TENSOR_KEYS:
        assert torch.allclose(getattr(frames, k), getattr(loaded, k), atol=1e-10)
    assert loaded.spatial_dim == 2
    assert abs(loaded.cycle_period - frames.cycle_period) < 1e-12
    assert abs(loaded.rho - frames.rho) < 1e-9 and abs(loaded.mu - frames.mu) < 1e-12


def test_manifest_csv_roundtrip(tmp_path):
    frames = _frames_2d()
    manifest = save_ibfe_manifest_csv(frames, tmp_path)
    loaded = load_ibfe_manifest(manifest)
    # Same total row counts (grouping by frame time is loss-less here).
    assert loaded.coords_fluid.shape == frames.coords_fluid.shape
    assert loaded.coords_wall.shape == frames.coords_wall.shape
    assert loaded.coords_mitral.shape[0] == frames.coords_mitral.shape[0]
    # CSV is written at 8 sig-figs; values should match closely (order may differ,
    # but per-frame selection preserves order within a frame).
    assert loaded.velocity_fluid.shape == frames.velocity_fluid.shape


def test_validator_passes_synthetic():
    frames = _frames_2d()
    report = validate_ibfe_frames(frames)
    assert report.ok, report.summary()
    assert report.stats["n_fluid"] > 0 and report.stats["n_mitral"] > 0


def test_validator_flags_bad_shapes_and_nans():
    frames = _frames_2d(with_valve=False)
    bad = dataclasses.replace(frames, velocity_fluid=frames.velocity_fluid[:, :1])
    r = validate_ibfe_frames(bad)
    assert not r.ok and any("velocity_fluid" in e for e in r.errors)

    nan_vel = frames.velocity_fluid.clone()
    nan_vel[0, 0] = float("nan")
    bad2 = dataclasses.replace(frames, velocity_fluid=nan_vel)
    r2 = validate_ibfe_frames(bad2)
    assert not r2.ok


def _frames_3d(with_valve=True):
    torch.set_default_dtype(torch.float64)
    cfg = Config()
    return synthetic_ibfe_frames_3d(cfg, n_fluid=500, n_wall=200, n_frames=4,
                                    with_valve=with_valve)


def test_3d_exporter_and_validation():
    frames = _frames_3d()
    assert frames.spatial_dim == 3
    assert frames.coords_fluid.shape[1] == 4
    assert frames.velocity_fluid.shape[1] == 3
    assert frames.traction_wall.shape[1] == 3
    report = validate_ibfe_frames(frames)
    assert report.ok, report.summary()


def test_3d_npz_and_manifest_roundtrip(tmp_path):
    frames = _frames_3d()
    # NPZ round-trip preserves every tensor + spatial_dim.
    npz = save_ibfe_npz(frames, tmp_path / "b3d.npz")
    ld = load_ibfe_npz(npz)
    assert ld.spatial_dim == 3
    for k in NPZ_TENSOR_KEYS:
        assert torch.allclose(getattr(frames, k), getattr(ld, k), atol=1e-9), k
    # Manifest/CSV round-trip (header-matched columns) reloads a valid 3D bundle.
    mp = save_ibfe_manifest_csv(frames, tmp_path / "dir3d")
    lm = load_ibfe_manifest(mp)
    assert lm.spatial_dim == 3
    assert lm.coords_fluid.shape[1] == 4 and lm.velocity_fluid.shape[1] == 3
    assert lm.normals_wall.shape[1] == 3 and lm.traction_wall.shape[1] == 3
    assert validate_ibfe_frames(lm).ok
    # Dispatch by path also yields a 3D bundle.
    fd = load_ibfe_output(str(tmp_path / "dir3d"), spatial_dim=3)
    assert fd.spatial_dim == 3 and fd.coords_fluid.shape[1] == 4


def test_3d_adapter_and_training_with_traction():
    torch.set_default_dtype(torch.float32)
    frames = _frames_3d(with_valve=True)
    wins = default_windows_from_frames(frames, n_windows=3)
    assert len(wins) == 3 and all(len(w) == 3 for w in wins)
    # Multi-window 3D Doppler is single-component (beam-projected) in 3D.
    dX, bd, vb = build_doppler_from_frames(frames, wins, noise_level=0.05, seed=0)
    assert dX.shape[1] == 4 and bd.shape[1] == 3 and vb.shape[1] == 1
    assert dX.shape[0] == 3 * frames.coords_fluid.shape[0]
    # End-to-end: FSI-informed backbone with 3D traction-continuity + valve + scalar.
    model, metrics = train_ibfe(frames, backbone="fsi_informed", steps=25,
                                lbfgs_iters=0, predict_scalar=True, use_traction=True,
                                transducers=wins, verbose=False)
    assert model.spatial_dim == 3
    for key in ("vel_relL2_u", "vel_relL2_v", "vel_relL2_w", "vel_relL2_speed",
                "pressure_relL2", "vorticity_absmax"):
        assert key in metrics and metrics[key] == metrics[key]  # finite


def test_load_ibfe_output_dispatch(tmp_path):
    frames = _frames_2d()
    npz = save_ibfe_npz(frames, tmp_path / "b.npz")
    save_ibfe_manifest_csv(frames, tmp_path / "dir")

    f_npz = load_ibfe_output(str(npz), spatial_dim=2)
    assert f_npz.coords_fluid.shape == frames.coords_fluid.shape
    f_dir = load_ibfe_output(str(tmp_path / "dir"), spatial_dim=2)
    assert f_dir.coords_fluid.shape[1] == 3

    # subsample caps points per set.
    f_sub = load_ibfe_output(str(npz), spatial_dim=2, subsample=100)
    assert f_sub.coords_fluid.shape[0] <= 100


def test_load_ibfe_output_bad_path(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        load_ibfe_output(str(tmp_path / "nope.foo"), spatial_dim=2)


def test_adapter_batches_and_training():
    torch.set_default_dtype(torch.float32)
    frames = _frames_2d(with_valve=True)
    builder = make_ibfe_batch_builder(frames, use_traction=True, predict_scalar=True,
                                      n_data=256, n_col=256, n_wall=128, n_valve=64)
    batch = builder(0)
    for key in ("data", "collocation", "wall", "traction", "valve", "scalar_inflow"):
        assert key in batch, key
    assert batch["data"]["beam_dir"].shape[1] == 2

    model, metrics = train_ibfe(frames, backbone="fsi_informed", steps=30,
                                lbfgs_iters=0, predict_scalar=True, use_traction=True,
                                verbose=False)
    assert model.predict_scalar
    for v in metrics.values():
        assert v == v  # finite


def test_ibfe_forcing_shuffle_decorrelates():
    torch.set_default_dtype(torch.float32)
    frames = _frames_2d()
    plain = make_ibfe_batch_builder(frames, seed=0, n_col=frames.coords_fluid.shape[0])
    shuf = make_ibfe_batch_builder(frames, seed=0, n_col=frames.coords_fluid.shape[0],
                                   shuffle_forcing=True)
    fp = plain(0)["collocation"]["forcing"]
    fs = shuf(0)["collocation"]["forcing"]
    assert fp.shape == fs.shape
    # Same marginal (a permutation) but not the same per-row ordering.
    assert torch.allclose(fp.sort(dim=0).values, fs.sort(dim=0).values, atol=1e-5)
    assert not torch.allclose(fp, fs)
    model, metrics = train_ibfe(frames, backbone="fsi_informed", steps=20,
                                lbfgs_iters=0, shuffle_forcing=True, verbose=False)
    for v in metrics.values():
        assert v == v
