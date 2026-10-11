"""Corrado LV-phantom NIfTI → public-volume NPZ."""

from pathlib import Path

import numpy as np
import pytest

nibabel = pytest.importorskip("nibabel")

from pinnecho.data.corrado_lv import (
    convert_corrado_scan, discover_mask, load_corrado_scan,
)
from pinnecho.data import has_oracle_forcing, load_ibfe_output


def _write_scan(root: Path, *, nx=8, ny=8, nz=10, nt=4,
                spacing_mm=(1.5, 1.5, 2.0), dt=0.05,
                origin_mm=(10.0, 20.0, 30.0),
                vx_mm_s=100, mask_4d=False) -> Path:
    """Day 1 / PC VIPR / 30_Minutes layout with int16 mm/s velocities."""
    scan = root / "Day 1" / "PC VIPR" / "30_Minutes"
    scan.mkdir(parents=True)
    affine = np.diag([spacing_mm[0], spacing_mm[1], spacing_mm[2], 1.0])
    affine[:3, 3] = origin_mm
    vx = np.zeros((nx, ny, nz, nt), dtype=np.int16)
    vy = np.zeros_like(vx)
    vz = np.zeros_like(vx)
    vx[..., 0] = vx_mm_s
    for stem, arr in (("VELX", vx), ("VELY", vy), ("VELZ", vz)):
        img = nibabel.Nifti1Image(arr, affine)
        img.header.set_data_dtype(np.int16)
        img.header.set_zooms((*spacing_mm, dt))
        nibabel.save(img, str(scan / f"{stem}.nii.gz"))
    mask = np.zeros((nx, ny, nz), dtype=np.uint8)
    mask[2:6, 2:6, 2:8] = 1
    if mask_4d:
        mask = np.repeat(mask[..., None], nt, axis=3)
        m_img = nibabel.Nifti1Image(mask, affine)
        m_img.header.set_data_dtype(np.uint8)
        m_img.header.set_zooms((*spacing_mm, dt))
        nibabel.save(m_img, str(scan.parent / "Mask_VIPR_Day1.nii.gz"))
    else:
        m_img = nibabel.Nifti1Image(mask, affine)
        m_img.header.set_data_dtype(np.uint8)
        m_img.header.set_zooms(spacing_mm)
        nibabel.save(m_img, str(scan.parent / "Mask_VIPR_Day1.nii.gz"))
    return scan


def test_discover_mask_vipr_day1(tmp_path):
    scan = _write_scan(tmp_path)
    found = discover_mask(scan)
    assert found.name == "Mask_VIPR_Day1.nii.gz"


def test_load_converts_mm_to_si(tmp_path):
    scan = _write_scan(tmp_path)
    vol = load_corrado_scan(scan)
    assert vol["velocity"].shape == (4, 8, 8, 10, 3)
    # 100 mm/s → 0.1 m/s on the first frame, u-component.
    assert vol["velocity"][0, 0, 0, 0, 0] == pytest.approx(0.1)
    assert vol["velocity"][0, 0, 0, 0, 1] == 0.0
    np.testing.assert_allclose(vol["spacing"], (0.0015, 0.0015, 0.002))
    np.testing.assert_allclose(vol["origin"], (0.01, 0.02, 0.03))
    np.testing.assert_allclose(vol["times"], (0.0, 0.05, 0.10, 0.15))
    assert vol["cycle_period"] == pytest.approx(0.20)
    assert vol["mask"].ndim == 3
    assert int(vol["mask"].sum()) == 4 * 4 * 6


def test_stride_keeps_physical_times(tmp_path):
    scan = _write_scan(tmp_path, nt=4, dt=0.05)
    vol = load_corrado_scan(scan, stride=2)
    assert vol["velocity"].shape[0] == 2
    np.testing.assert_allclose(vol["times"], (0.0, 0.10))
    assert vol["cycle_period"] == pytest.approx(0.20)


def test_convert_roundtrip_is_public_volume(tmp_path):
    scan = _write_scan(tmp_path)
    out = tmp_path / "day1_vipr_30min.npz"
    path, _ = convert_corrado_scan(scan, out)
    frames = load_ibfe_output(str(path), n_fluid=200, n_wall=40)
    assert frames.spatial_dim == 3
    assert not has_oracle_forcing(frames)
    assert float(frames.forcing_fluid.abs().max()) == 0.0
    assert float(frames.pressure_fluid.abs().max()) == 0.0
    assert frames.coords_fluid.shape[0] > 0


def test_piv_mask_name(tmp_path):
    scan = tmp_path / "Registered to Day 1" / "PC VIPR" / "Average Beat"
    scan.mkdir(parents=True)
    arr = np.zeros((4, 4, 4, 2), dtype=np.int16)
    aff = np.eye(4)
    for stem in ("VELX", "VELY", "VELZ"):
        img = nibabel.Nifti1Image(arr, aff)
        img.header.set_zooms((2.0, 2.0, 2.0, 0.04))
        nibabel.save(img, str(scan / f"{stem}.nii.gz"))
    mask = np.ones((4, 4, 4), dtype=np.uint8)
    nibabel.save(nibabel.Nifti1Image(mask, aff),
                 str(scan.parent / "MASK.nii.gz"))
    assert discover_mask(scan).name == "MASK.nii.gz"
    vol = load_corrado_scan(scan)
    assert vol["velocity"].shape == (2, 4, 4, 4, 3)


def test_missing_component_raises(tmp_path):
    scan = _write_scan(tmp_path)
    (scan / "VELY.nii.gz").unlink()
    with pytest.raises(FileNotFoundError, match="VELY"):
        load_corrado_scan(scan)
