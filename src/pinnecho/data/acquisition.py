"""Multi-plane echocardiographic acquisition geometry.

A transthoracic echo exam does not sample the whole left-ventricular volume. It
captures a handful of standard 2D *imaging planes* through the heart -- apical
4-chamber (A4C), apical 2-chamber (A2C), parasternal long-axis (PLAX) and
parasternal short-axis (PSAX) -- each from a fixed probe position (the apex of a
sector scan). Two physical facts dominate what a Doppler PINN can observe:

1. **Single component.** Doppler only resolves the velocity component *along the
   beam* (apex -> point), so each view measures a projection, never the full
   vector (handled already by :mod:`pinnecho.data.synthesize_doppler`).
2. **Plane coverage.** A 2D view only insonifies tissue lying in (a thin slab
   around) its imaging plane; the out-of-plane flow is simply not seen. With
   only a few standard planes the angular + spatial coverage is sparse -- this
   is the structural reason 3D intraventricular flow reconstruction from Doppler
   is hard, and the observability result lives here, not in SNR or point count.

This module adds the plane-coverage layer on top of the beam-projection /
Nyquist / SNR synthesiser:

* :class:`ImagingPlane` -- a probe apex + unit slab normal (+ optional sector
  half-angle and max depth); ``select_mask`` keeps the fluid points a view can
  actually insonify.
* :func:`standard_views_from_extents` / :func:`standard_views_from_frames` --
  the canonical A4C/A2C/PLAX/PSAX set sized to a cavity's centre and extents.
* :func:`synthesize_multiplane_doppler` -- per-view masking + beam projection,
  then the shared sparsity/aliasing/SNR tail (reproducible with
  :func:`~pinnecho.data.synthesize_doppler.synthesize_doppler`).
* :func:`build_multiplane_doppler_from_frames` -- a drop-in replacement for
  :func:`~pinnecho.data.ibfe_dataset.build_doppler_from_frames` that feeds the
  IBFE training adapter with plane-limited data.

In 2D the imaging plane is the whole domain (echo *is* a 2D image): a plane with
``normal=None`` performs no slab filtering, so a 2D "view" reduces to the plain
point-window behaviour and this stays dimension-generic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from .synthesize_doppler import (
    DopplerMeasurements,
    beam_directions,
    _finalize_measurements,
)

# Canonical standard-view names per dimension.
VIEWS_3D = ("a4c", "a2c", "plax", "psax")
VIEWS_2D = ("apical", "lateral")


@dataclass
class ImagingPlane:
    """One echocardiographic imaging view.

    ``apex`` is the probe origin (``(dim,)``) from which beams emanate.
    ``normal`` is the unit slab normal (``(dim,)``): a point is in-plane when its
    signed distance to the plane through ``apex`` is within ``thickness`` (the
    elevation / slice half-thickness, metres). ``normal=None`` disables the slab
    filter -- the whole domain is in-plane, the natural 2D behaviour. Optionally
    ``sector_halfangle`` (radians, about ``axis``) and ``max_depth`` (metres from
    the apex) bound the sector field of view.
    """

    name: str
    apex: torch.Tensor
    normal: Optional[torch.Tensor] = None
    thickness: float = 5e-3
    sector_halfangle: Optional[float] = None
    axis: Optional[torch.Tensor] = None
    max_depth: Optional[float] = None

    @property
    def dim(self) -> int:
        return int(self.apex.shape[-1])

    def _rel(self, coords: torch.Tensor) -> torch.Tensor:
        d = self.dim
        return coords[:, :d] - self.apex.to(coords.dtype).reshape(1, d)

    def select_mask(self, coords: torch.Tensor) -> torch.Tensor:
        """Boolean mask of the points this view can insonify (``(N,)``)."""
        rel = self._rel(coords)
        mask = torch.ones(coords.shape[0], dtype=torch.bool, device=coords.device)
        if self.normal is not None:
            n = self.normal.to(coords.dtype).reshape(-1)
            n = n / n.norm().clamp_min(1e-12)
            dist = (rel * n.reshape(1, -1)).sum(dim=1).abs()
            mask &= dist <= self.thickness
        if self.max_depth is not None:
            mask &= rel.norm(dim=1) <= self.max_depth
        if self.sector_halfangle is not None and self.axis is not None:
            a = self.axis.to(coords.dtype).reshape(-1)
            a = a / a.norm().clamp_min(1e-12)
            cosang = (rel * a.reshape(1, -1)).sum(dim=1) / rel.norm(dim=1).clamp_min(1e-12)
            mask &= cosang.clamp(-1.0, 1.0) >= math.cos(self.sector_halfangle)
        return mask

    def beams(self, coords: torch.Tensor) -> torch.Tensor:
        """Unit beam directions (apex -> point) at ``coords`` (``(N, dim)``)."""
        return beam_directions(coords, self.apex.to(coords.dtype))


def _vec(values, dtype) -> torch.Tensor:
    return torch.tensor(list(values), dtype=dtype)


def standard_views_from_extents(
    center: Sequence[float],
    extents: Sequence[float],
    dim: int,
    thickness: float = 5e-3,
    views: Optional[Sequence[str]] = None,
    apex_dist: float = 2.5,
    dtype: Optional[torch.dtype] = None,
) -> List[ImagingPlane]:
    """Build the standard echo view set for a cavity.

    ``center`` / ``extents`` are per-axis cavity centre and full ranges (metres);
    the long (apex-base) axis is ``z`` in 3D (index 2) following the project's
    geometry convention. Probe apices sit ``apex_dist * extent`` outside the
    cavity along each view direction.

    3D views (``views`` subset of :data:`VIEWS_3D`):
      * ``a4c``  apical, probe below (-z); slab normal +y (the x-z plane).
      * ``a2c``  apical, same probe; slab normal +x (the y-z plane, A4C rotated).
      * ``plax`` parasternal (probe +x); slab normal +y (long-axis plane).
      * ``psax`` parasternal (probe +x); slab normal +z (short-axis cross-section).

    2D views (``views`` subset of :data:`VIEWS_2D`) have ``normal=None`` (no slab,
    the whole 2D image) and only differ in apex -> beam angle:
      * ``apical``  probe below (-y).   * ``lateral`` probe to the side (+x).
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    c = list(center)
    e = list(extents)
    if dim == 3:
        apex_apical = (c[0], c[1], c[2] - apex_dist * e[2])
        apex_para = (c[0] + apex_dist * e[0], c[1], c[2])
        spec = {
            "a4c": (apex_apical, (0.0, 1.0, 0.0)),
            "a2c": (apex_apical, (1.0, 0.0, 0.0)),
            "plax": (apex_para, (0.0, 1.0, 0.0)),
            "psax": (apex_para, (0.0, 0.0, 1.0)),
        }
        chosen = tuple(views) if views is not None else VIEWS_3D
        planes = []
        for name in chosen:
            apex, normal = spec[name]
            planes.append(ImagingPlane(
                name=name, apex=_vec(apex, dtype), normal=_vec(normal, dtype),
                thickness=float(thickness)))
        return planes
    if dim == 2:
        spec2 = {
            "apical": (c[0], c[1] - apex_dist * e[1]),
            "lateral": (c[0] + apex_dist * e[0], c[1]),
        }
        chosen = tuple(views) if views is not None else VIEWS_2D
        return [ImagingPlane(name=name, apex=_vec(spec2[name], dtype), normal=None)
                for name in chosen]
    raise ValueError(f"standard_views_from_extents supports dim 2 or 3, got {dim}")


def standard_views_from_frames(
    frames,
    views: Optional[Sequence[str]] = None,
    thickness_frac: float = 0.12,
    apex_dist: float = 2.5,
) -> List[ImagingPlane]:
    """Standard echo views sized to an :class:`IBFEFrames` fluid point cloud.

    The slab half-thickness is ``thickness_frac`` of the cavity's mean spatial
    extent (so ~12% of a ~4 cm cavity is a few-mm slice, a realistic elevation
    resolution). Returns :class:`ImagingPlane` objects in ``frames`` dtype.
    """
    dim = frames.spatial_dim
    coords = frames.coords_fluid[:, :dim]
    center = coords.mean(dim=0)
    ext = coords.max(dim=0).values - coords.min(dim=0).values
    thickness = float(thickness_frac) * float(ext.mean())
    return standard_views_from_extents(
        center.tolist(), ext.tolist(), dim, thickness=thickness, views=views,
        apex_dist=apex_dist, dtype=coords.dtype)


def synthesize_multiplane_doppler(
    coords: torch.Tensor,
    velocity: torch.Tensor,
    planes: Sequence[ImagingPlane],
    v_nyquist: float = 0.8,
    snr_db: float = 20.0,
    sparsity: float = 0.5,
    dealias: bool = False,
    rng: Optional[torch.Generator] = None,
) -> DopplerMeasurements:
    """Multi-plane Doppler: each view only sees points inside its imaging plane.

    Unlike :func:`~pinnecho.data.synthesize_doppler.synthesize_doppler` (every
    point visible from every window), each :class:`ImagingPlane` first masks the
    fluid points it can insonify, then projects velocity onto its beam. The
    pooled, per-view-masked samples get the shared sparsity/aliasing/SNR tail.
    ``window_id`` indexes into ``planes``. ``coords`` is ``(N, >=dim)`` (extra
    columns such as time are carried through); ``velocity`` is ``(N, >=dim)``.
    """
    if len(planes) == 0:
        raise ValueError("synthesize_multiplane_doppler needs at least one plane")
    dim = planes[0].dim
    coords_list, beam_list, vclean_list, win_list = [], [], [], []
    for k, plane in enumerate(planes):
        mask = plane.select_mask(coords)
        if int(mask.sum()) == 0:
            continue
        c = coords[mask]
        v = velocity[mask]
        beam = plane.beams(c)
        v_beam = (v[:, :dim] * beam).sum(dim=1, keepdim=True)
        coords_list.append(c)
        beam_list.append(beam)
        vclean_list.append(v_beam)
        win_list.append(torch.full((c.shape[0], 1), k, dtype=torch.long))
    if not coords_list:
        raise ValueError(
            "No fluid points fell inside any imaging plane; increase the slab "
            "thickness (thickness_frac) or check the view/cavity geometry.")

    coords_sel = torch.cat(coords_list, dim=0)
    beam = torch.cat(beam_list, dim=0)
    v_clean = torch.cat(vclean_list, dim=0)
    window_id = torch.cat(win_list, dim=0)

    return _finalize_measurements(
        coords_sel, beam, v_clean, window_id,
        v_nyquist=v_nyquist, snr_db=snr_db, sparsity=sparsity,
        dealias=dealias, rng=rng,
    )


def build_multiplane_doppler_from_frames(
    frames,
    planes: Sequence[ImagingPlane],
    noise_level: float = 0.05,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Plane-limited ``(data_X, beam_dir, v_beam)`` for the IBFE training adapter.

    Drop-in replacement for
    :func:`~pinnecho.data.ibfe_dataset.build_doppler_from_frames` that applies
    the :class:`ImagingPlane` slab masks: only the fluid points a view can
    insonify contribute data for that view. ``data_X`` keeps the full
    ``(x[, y, z], t)`` row so the model sees space-time inputs; noise is relative
    to the mean |v_beam| (matching the IBFE adapter's convention).
    """
    dim = frames.spatial_dim
    g = torch.Generator().manual_seed(seed)
    X = frames.coords_fluid
    vel = frames.velocity_fluid
    Xs, beams, vs = [], [], []
    for plane in planes:
        mask = plane.select_mask(X)
        if int(mask.sum()) == 0:
            continue
        Xp = X[mask]
        b = plane.beams(Xp)
        vb = (vel[mask][:, :dim] * b).sum(dim=1, keepdim=True)
        if noise_level > 0:
            vb = vb + noise_level * vb.abs().mean() * torch.randn(
                vb.shape, generator=g, dtype=vb.dtype)
        Xs.append(Xp.clone())
        beams.append(b)
        vs.append(vb)
    if not Xs:
        raise ValueError(
            "No fluid points fell inside any imaging plane; increase the slab "
            "thickness (thickness_frac) or check the view/cavity geometry.")
    return torch.cat(Xs, 0), torch.cat(beams, 0), torch.cat(vs, 0)
