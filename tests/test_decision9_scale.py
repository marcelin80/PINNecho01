"""Decision 9 approximate V0: isotropic scale of a reference mesh."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from pinnecho.config import Config
from pinnecho.data.decision9_scale import (
    Decision9ScaleError,
    apply_isotropic_v0,
    isotropic_scale_points,
    load_vertices,
    refuse_if_ibfe_export,
    tag_ibfe_export_approx,
    volume_ml,
    write_scaled_mesh,
)
from pinnecho.data.ibfe_fidelity import (
    IMAGE_VOLUME_ML_PAT12,
    KLOTZ_V0_ML,
    V0_SCALE_ISOTROPIC,
    diastole_usable,
    isotropic_linear_scale,
    resolve_fidelity,
)
from pinnecho.data.ibfe_io import save_ibfe_npz
from pinnecho.data.load_ibfe_output import load_ibfe_output, synthetic_ibfe_frames


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "decision9_isotropic_v0.py"


def _cube(side: float = 0.05):
    """Axis-aligned cube [0, side]^3 in metres (default 125 mL)."""
    s = float(side)
    verts = s * np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ])
    faces = np.array([
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [0, 4, 7], [0, 7, 3],
        [1, 2, 6], [1, 6, 5],
    ], dtype=int)
    return verts, faces


def test_cube_volume_and_scale_sets_v0():
    verts, faces = _cube(0.05)
    assert abs(volume_ml(verts, faces) - 125.0) < 1e-6
    scaled, report = apply_isotropic_v0(
        verts, v0_ml=58.0, image_volume_ml=125.0, faces=faces,
    )
    expect = isotropic_linear_scale(58.0, 125.0)
    assert abs(report.scale - expect) < 1e-12
    assert abs(report.measured_v0_ml - 58.0) < 1e-6
    assert report.approximation is True
    # Linear sizes shrink by the same factor; centroid stays put.
    assert np.allclose(scaled.mean(axis=0), verts.mean(axis=0), atol=1e-12)
    assert np.allclose(scaled.max(axis=0) - scaled.min(axis=0),
                       expect * (verts.max(axis=0) - verts.min(axis=0)))


def test_default_pat12_scale_is_exposed():
    _, report = apply_isotropic_v0(
        np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]),
        v0_ml=KLOTZ_V0_ML,
        image_volume_ml=IMAGE_VOLUME_ML_PAT12,
    )
    assert abs(report.scale - V0_SCALE_ISOTROPIC) < 1e-12
    assert abs(report.scale - 0.82) < 0.01
    assert report.p_img_mmhg == (5.0, 10.0)
    assert abs(report.wall_thickness_approx_mm - 7.4 * report.scale) < 1e-12


def test_scale_about_explicit_center():
    pts = np.array([[2.0, 0.0], [4.0, 0.0]])
    out = isotropic_scale_points(pts, 0.5, center=np.array([0.0, 0.0]))
    assert np.allclose(out, [[1.0, 0.0], [2.0, 0.0]])


def test_refuse_ibfe_export(tmp_path):
    torch.set_default_dtype(torch.float64)
    frames = synthetic_ibfe_frames(Config(), n_fluid=40, n_wall=20, n_frames=2)
    path = save_ibfe_npz(frames, tmp_path / "beat.npz")
    try:
        refuse_if_ibfe_export(path)
        raise AssertionError("expected Decision9ScaleError")
    except Decision9ScaleError as exc:
        assert "simulated IBFEFrames" in str(exc)
    try:
        load_vertices(path)
        raise AssertionError("expected Decision9ScaleError")
    except Decision9ScaleError:
        pass


def test_write_sidecar_and_tag_export(tmp_path):
    verts, faces = _cube(0.05)
    scaled, report = apply_isotropic_v0(
        verts, v0_ml=58.0, image_volume_ml=125.0, faces=faces,
    )
    out = write_scaled_mesh(tmp_path / "v0", scaled, report, faces=faces)
    fid = json.loads((out / "fidelity.json").read_text())
    assert fid["decision_9"] == "approx"
    assert fid["approximation"] is True
    assert fid["diastole_usable"] is True
    assert "v0_scale" in fid
    assert (out / "vertices_scaled.csv").exists()
    assert "approximation" in (out / "report.md").read_text().lower()

    torch.set_default_dtype(torch.float64)
    frames = synthetic_ibfe_frames(Config(), n_fluid=40, n_wall=20, n_frames=2)
    tagged = tag_ibfe_export_approx(frames, v0_ml=58.0, image_volume_ml=125.0,
                                    p_img_mmhg=(5.0,))
    assert resolve_fidelity(tagged).decision_9 == "approx"
    assert diastole_usable(tagged) is True
    assert abs(tagged.fidelity.v0_scale - isotropic_linear_scale(58.0, 125.0)) < 1e-12


def test_cli_print_scale_and_scale(tmp_path):
    printed = subprocess.check_output(
        [sys.executable, str(SCRIPT), "print-scale",
         "--v0-ml", "58", "--image-volume-ml", "105"],
        text=True,
    )
    payload = json.loads(printed)
    assert abs(payload["v0_scale"] - V0_SCALE_ISOTROPIC) < 1e-12

    verts, faces = _cube(0.05)
    vpath = tmp_path / "verts.csv"
    fpath = tmp_path / "faces.csv"
    np.savetxt(vpath, verts, delimiter=",", header="x,y,z", comments="")
    np.savetxt(fpath, faces, delimiter=",", header="i,j,k", comments="", fmt="%d")
    out = tmp_path / "scaled"
    subprocess.check_call(
        [sys.executable, str(SCRIPT), "scale",
         "--vertices", str(vpath), "--faces", str(fpath),
         "--v0-ml", "58", "--image-volume-ml", "125",
         "--out", str(out)],
    )
    report = json.loads((out / "report.json").read_text())
    assert abs(report["measured_v0_ml"] - 58.0) < 1e-5


def test_cli_missing_vertices_is_clear():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "scale",
         "--vertices", "path/to/endo_vertices.csv",
         "--out", "/tmp/decision9_nope"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "not found" in proc.stderr
    assert "python3" in proc.stderr


def test_cli_refuses_ibfe_export(tmp_path):
    torch.set_default_dtype(torch.float64)
    frames = synthetic_ibfe_frames(Config(), n_fluid=40, n_wall=20, n_frames=2)
    path = save_ibfe_npz(frames, tmp_path / "beat.npz")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "scale",
         "--vertices", str(path), "--out", str(tmp_path / "nope")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "simulated IBFEFrames" in proc.stderr


def test_cli_tag_export(tmp_path):
    torch.set_default_dtype(torch.float64)
    frames = synthetic_ibfe_frames(Config(), n_fluid=40, n_wall=20, n_frames=2)
    src = save_ibfe_npz(frames, tmp_path / "beat.npz")
    dst = tmp_path / "beat_d9.npz"
    subprocess.check_call(
        [sys.executable, str(SCRIPT), "tag",
         "--export", str(src), "--out", str(dst),
         "--v0-ml", "58", "--image-volume-ml", "105",
         "--p-img-mmhg", "5"],
    )
    loaded = load_ibfe_output(str(dst))
    fid = resolve_fidelity(loaded)
    assert fid.decision_9 == "approx"
    assert fid.p_img_mmhg == (5.0,)
    assert diastole_usable(loaded) is True
