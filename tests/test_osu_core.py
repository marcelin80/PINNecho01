"""OSU CORe phase → m/s packing (no MATLAB, no Zenodo download)."""

import numpy as np

from pinnecho.data.osu_core import (
    DATASET11_VENC_MPS, magnitude_mask, pack_osu_public_volume,
    phase_to_velocity_mps, velocity_from_thetas,
)
from pinnecho.data.public_volume import save_public_volume_npz, load_public_volume_npz
from pinnecho.data import has_oracle_forcing


def test_phase_pi_is_venc():
    v = phase_to_velocity_mps(np.pi, DATASET11_VENC_MPS)
    assert v == 2.0


def test_velocity_from_matlab_xyzt():
    nx = ny = nz = 4
    t_len = 5
    # last axis is time
    tx = np.zeros((nx, ny, nz, t_len))
    tx[..., 2] = np.pi  # frame 2 is +VENC in x
    vel = velocity_from_thetas(tx, np.zeros_like(tx), np.zeros_like(tx), 2.0)
    assert vel.shape == (t_len, nx, ny, nz, 3)
    assert vel[2, ..., 0].max() == 2.0
    assert vel[0].max() == 0.0


def test_pack_has_no_oracle_f(tmp_path):
    n, t_len = 6, 4
    th = np.zeros((n, n, n, t_len))
    th[:2, :, :, :] = 0.5 * np.pi
    xhat = np.ones((n, n, n, t_len))
    xhat[:3] = 4.0
    vol = pack_osu_public_volume(
        {"thetaX": th, "thetaY": th * 0, "thetaZ": th * 0, "xHat": xhat},
        venc_mps=2.0, spacing_m=(0.002, 0.002, 0.002), period_s=0.8)
    assert vol["velocity"].shape == (t_len, n, n, n, 3)
    assert vol["mask"].any()
    assert not vol["mask_is_lv"]
    path = save_public_volume_npz(
        tmp_path / "osu.npz", velocity=vol["velocity"], mask=vol["mask"],
        spacing=vol["spacing"], origin=vol["origin"], times=vol["times"],
        cycle_period=0.8, rho=1060.0, mu=0.0035)
    frames = load_public_volume_npz(path, n_fluid=80, n_wall=20)
    assert not has_oracle_forcing(frames)


def test_magnitude_mask_is_not_empty():
    xhat = np.zeros((5, 5, 5, 3))
    xhat[2:, 2:, 2:] = 10.0
    m = magnitude_mask(xhat, percentile=80.0)
    assert m.any()
    assert not m.all()
