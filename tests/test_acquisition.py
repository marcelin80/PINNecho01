"""Tests for multi-plane echo acquisition geometry (imaging planes/views)."""

import math

import pytest
import torch

from pinnecho.config import Config
from pinnecho.data.acquisition import (
    ImagingPlane,
    VIEWS_2D,
    VIEWS_3D,
    build_multiplane_doppler_from_frames,
    standard_views_from_extents,
    standard_views_from_frames,
    synthesize_multiplane_doppler,
)
from pinnecho.data.load_ibfe_output import (
    synthetic_ibfe_frames,
    synthetic_ibfe_frames_3d,
)
from pinnecho.data.synthesize_doppler import synthesize_doppler
from pinnecho.data.ibfe_dataset import train_ibfe

torch.set_default_dtype(torch.float64)


def test_standard_views_3d_names_and_geometry():
    planes = standard_views_from_extents([0.0, 0.0, 0.0], [0.04, 0.04, 0.06], 3,
                                         thickness=0.006)
    assert [p.name for p in planes] == list(VIEWS_3D)
    by = {p.name: p for p in planes}
    # Apical views share a probe below the cavity (-z); parasternal to the side (+x).
    assert by["a4c"].apex[2] < 0 and torch.allclose(by["a4c"].apex, by["a2c"].apex)
    assert by["plax"].apex[0] > 0 and torch.allclose(by["plax"].apex, by["psax"].apex)
    # A4C/A2C slabs are orthogonal; PSAX cross-section normal is the long axis z.
    assert by["a4c"].normal.tolist() == [0.0, 1.0, 0.0]
    assert by["a2c"].normal.tolist() == [1.0, 0.0, 0.0]
    assert by["psax"].normal.tolist() == [0.0, 0.0, 1.0]


def test_standard_views_2d_have_no_slab():
    planes = standard_views_from_extents([0.0, 0.0], [0.04, 0.06], 2)
    assert [p.name for p in planes] == list(VIEWS_2D)
    assert all(p.normal is None for p in planes)       # 2D image = whole domain
    assert planes[0].apex[1] < 0 and planes[1].apex[0] > 0


def test_slab_selection_keeps_in_plane_points():
    # A4C slab normal +y, thickness 6 mm: only |y| <= thickness survives.
    plane = ImagingPlane(name="a4c", apex=torch.tensor([0.0, 0.0, -0.1]),
                         normal=torch.tensor([0.0, 1.0, 0.0]), thickness=0.006)
    coords = (torch.rand(4000, 3) - 0.5) * 0.04       # +/-2 cm cube
    mask = plane.select_mask(coords)
    assert mask.any() and not mask.all()               # some in, some out
    assert float(coords[mask][:, 1].abs().max()) <= 0.006 + 1e-9
    # Points outside the slab are excluded.
    assert float(coords[~mask][:, 1].abs().min()) > 0.006 - 1e-9


def test_disjoint_slabs_select_different_points():
    coords = (torch.rand(3000, 3) - 0.5) * 0.04
    a4c = ImagingPlane("a4c", torch.tensor([0., 0., -0.1]),
                       torch.tensor([0., 1., 0.]), thickness=0.004)
    a2c = ImagingPlane("a2c", torch.tensor([0., 0., -0.1]),
                       torch.tensor([1., 0., 0.]), thickness=0.004)
    m1, m2 = a4c.select_mask(coords), a2c.select_mask(coords)
    # The two orthogonal slabs are far from identical selections.
    overlap = (m1 & m2).sum().item()
    assert overlap < 0.5 * min(m1.sum().item(), m2.sum().item())


def test_beams_are_unit_and_point_from_apex():
    plane = ImagingPlane("v", torch.tensor([0.0, 0.0, -1.0]))
    coords = torch.rand(50, 3)
    b = plane.beams(coords)
    assert torch.allclose(b.norm(dim=1), torch.ones(50), atol=1e-9)
    # beam should correlate with (point - apex) direction
    rel = coords[:, :3] - plane.apex
    rel = rel / rel.norm(dim=1, keepdim=True)
    assert torch.allclose(b, rel, atol=1e-9)


def test_multiplane_single_plane_matches_point_window():
    # A 2D plane with normal=None (whole domain) must equal a single point window.
    coords = torch.rand(300, 3)
    vel = torch.randn(300, 2)
    apex = torch.tensor([0.0, -2.0])
    plane = ImagingPlane("apical", apex, normal=None)
    g1 = torch.Generator().manual_seed(7)
    mp = synthesize_multiplane_doppler(coords, vel, [plane], v_nyquist=10.0,
                                       snr_db=100.0, sparsity=0.0, dealias=True, rng=g1)
    g2 = torch.Generator().manual_seed(7)
    sd = synthesize_doppler(coords, vel, apex.unsqueeze(0), v_nyquist=10.0,
                            snr_db=100.0, sparsity=0.0, dealias=True, rng=g2)
    assert torch.allclose(mp.v_beam_clean, sd.v_beam_clean, atol=1e-9)


def test_multiplane_window_ids_and_coverage():
    coords = (torch.rand(5000, 4) - 0.5) * 0.04        # x,y,z,t
    vel = torch.randn(5000, 3) * 0.3
    planes = standard_views_from_extents([0, 0, 0], [0.04, 0.04, 0.06], 3,
                                         thickness=0.006)
    m = synthesize_multiplane_doppler(coords, vel, planes, v_nyquist=10.0,
                                      snr_db=100.0, sparsity=0.0, dealias=True)
    # Each plane sees only a slab, so the pooled count is far below len*n.
    assert m.v_beam.shape[0] < 4 * coords.shape[0]
    assert set(m.window_id.unique().tolist()).issubset({0, 1, 2, 3})
    # Carried-through coordinate columns (incl. time) are preserved.
    assert m.coords_meas.shape[1] == 4


def test_empty_slab_raises():
    coords = torch.zeros(10, 3) + torch.tensor([0.0, 0.5, 0.0])  # all at y=0.5
    vel = torch.randn(10, 3)
    plane = ImagingPlane("a4c", torch.tensor([0., 0., -1.]),
                         torch.tensor([0., 1., 0.]), thickness=1e-4)
    with pytest.raises(ValueError):
        synthesize_multiplane_doppler(coords, vel, [plane], sparsity=0.0)


def test_standard_views_from_frames_scales_thickness():
    cfg = Config()
    frames = synthetic_ibfe_frames_3d(cfg, n_fluid=800, n_wall=200, n_frames=3)
    planes = standard_views_from_frames(frames, thickness_frac=0.1)
    ext = (frames.coords_fluid[:, :3].max(0).values
           - frames.coords_fluid[:, :3].min(0).values)
    expected = 0.1 * float(ext.mean())
    assert all(abs(p.thickness - expected) < 1e-12 for p in planes)


def test_frames_adapter_shapes_3d():
    cfg = Config()
    frames = synthetic_ibfe_frames_3d(cfg, n_fluid=1000, n_wall=200, n_frames=3)
    planes = standard_views_from_frames(frames, thickness_frac=0.18)
    dX, bd, vb = build_multiplane_doppler_from_frames(frames, planes,
                                                      noise_level=0.0, seed=0)
    assert dX.shape[1] == 4 and bd.shape[1] == 3 and vb.shape[1] == 1
    assert dX.shape[0] == bd.shape[0] == vb.shape[0] > 0


def test_multiplane_training_end_to_end_3d():
    cfg = Config()
    frames = synthetic_ibfe_frames_3d(cfg, n_fluid=900, n_wall=220, n_frames=3,
                                      with_valve=True)
    planes = standard_views_from_frames(frames, thickness_frac=0.2)
    model, metrics = train_ibfe(frames, backbone="fsi_informed", steps=25,
                                lbfgs_iters=3, planes=planes, use_traction=True,
                                predict_scalar=True, verbose=False)
    for k in ("vel_relL2_u", "vel_relL2_v", "vel_relL2_w", "pressure_relL2",
              "vorticity_absmax"):
        assert k in metrics and math.isfinite(metrics[k])


def test_frames_adapter_2d_still_works():
    cfg = Config()
    frames = synthetic_ibfe_frames(cfg, n_fluid=600, n_wall=150, n_frames=3)
    planes = standard_views_from_frames(frames)  # 2D: no slab, whole domain
    dX, bd, vb = build_multiplane_doppler_from_frames(frames, planes,
                                                      noise_level=0.0, seed=0)
    # Two 2D views see all points each => 2 * N_fluid rows.
    assert dX.shape[1] == 3 and bd.shape[1] == 2
    assert dX.shape[0] == 2 * frames.coords_fluid.shape[0]
