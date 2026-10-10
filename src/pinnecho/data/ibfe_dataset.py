"""Turn a real (or synthetic) :class:`IBFEFrames` bundle into a training run.

This is the drop-in that makes an IBAMR/IBFE export flow into the existing spec
trainer without touching the loss or optimisation loop:

* :func:`make_ibfe_batch_builder` -- a ``build_batches(step)`` closure producing
  the dict :class:`~pinnecho.train.composite_loss.CompositeLoss` consumes
  (Doppler ``data``, ``collocation`` + FSI ``forcing``, ``wall`` velocity,
  optional ``traction`` continuity, ``valve`` Dirichlet, ``scalar_inflow`` for
  residence time).
* :func:`build_model_for_ibfe` -- the shared :class:`PINNNet` + a backbone-
  appropriate :class:`CompositeLoss` (2D or 3D, sized from the data).
* :func:`evaluate_ibfe` -- held-out velocity / pressure / vorticity error vs the
  fluid ground truth.
* :func:`train_ibfe` -- a convenience wiring the above to the Adam(+L-BFGS) loop.

Only the beam-projected velocity component is used as data (Doppler); the full
``velocity_fluid`` is treated as held-out ground truth for evaluation.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..models.mlp_pinn import PINNNet
from ..train.composite_loss import AnnealSchedule, CompositeLoss, LossWeights
from ..train.train import TrainConfig, train_model
from .. import evaluate as ev


def input_bounds_from_frames(frames, pad: float = 0.08) -> Tuple[List[float], List[float]]:
    """Per-column ``(lows, highs)`` (spatial + time) from all point sets."""
    dim = frames.spatial_dim
    coords = [frames.coords_fluid, frames.coords_wall,
              frames.coords_mitral, frames.coords_aortic]
    coords = [c for c in coords if c.shape[0] > 0]
    allc = torch.cat(coords, dim=0)
    lo = allc.min(dim=0).values
    hi = allc.max(dim=0).values
    span = (hi - lo).clamp_min(1e-9)
    lo = lo - pad * span
    hi = hi + pad * span
    lo[dim] = min(0.0, float(lo[dim]))  # time starts at 0
    return lo.tolist(), hi.tolist()


def scales_from_frames(frames) -> Dict[str, float]:
    """Reference velocity / pressure / length scales from the data."""
    dim = frames.spatial_dim
    speed = frames.velocity_fluid.norm(dim=1)
    velocity = float(torch.quantile(speed, 0.99)) if speed.numel() else 1.0
    velocity = max(velocity, 1e-6)
    p = frames.pressure_fluid
    p_range = float(p.max() - p.min()) if p.numel() else 0.0
    pressure = max(p_range, frames.rho * velocity ** 2, 1.0)
    coords = frames.coords_fluid[:, :dim]
    length = float((coords.max(dim=0).values - coords.min(dim=0).values).mean()) * 0.5
    length = max(length, 1e-4)
    return {"velocity": velocity, "pressure": pressure, "length": length}


def default_windows_from_frames(frames, n_windows: int = 2) -> List[Tuple[float, ...]]:
    """Virtual transducer positions outside the cavity (apical + lateral)."""
    dim = frames.spatial_dim
    coords = frames.coords_fluid[:, :dim]
    center = coords.mean(dim=0)
    ext = (coords.max(dim=0).values - coords.min(dim=0).values)
    c = center.tolist()
    e = ext.tolist()
    if dim == 2:
        apical = (c[0], c[1] - 2.5 * e[1])
        lateral = (c[0] + 2.5 * e[0], c[1])
        return [apical, lateral][:max(1, n_windows)]
    apical = (c[0], c[1], c[2] - 2.5 * e[2])
    lat_x = (c[0] + 2.5 * e[0], c[1], c[2])
    lat_y = (c[0], c[1] + 2.5 * e[1], c[2])
    return [apical, lat_x, lat_y][:max(1, n_windows)]


def forcing_band_mask(forcing: torch.Tensor, tol: float = 0.0) -> torch.Tensor:
    """Boolean mask of points inside the forcing support (the IB wall band).

    Real IBFE forcing is the structural Lagrangian force spread to the Euler grid
    by the IB kernel, so it is nonzero only in a ~3-cell shell around the wall and
    exactly zero in the cavity interior. This returns ``‖f‖ > tol`` per point.
    """
    return forcing.norm(dim=1) > tol


def band_coverage_fraction(forcing: torch.Tensor, tol: float = 0.0) -> float:
    """Fraction of fluid points inside the forcing band (``‖f‖ > tol``).

    For real IB forcing this is the share of the sampled cavity volume where the
    FSI term is active; outside it Model B equals Model A. (At the coarse Minimum
    Goal resolution the real band covers only ~43-59% of the cavity.)
    """
    if forcing.shape[0] == 0:
        return 0.0
    return float(forcing_band_mask(forcing, tol=tol).float().mean())


def apply_forcing_control(forcing: torch.Tensor, coords: torch.Tensor,
                          mode: Optional[str], seed: int, dim: int) -> torch.Tensor:
    """Diagnostic forcing field for the **oracle** forcing experiment (see below).

    The forcing ``f`` is evaluation-only ground truth, never a deliverable Model B
    input (it is unavailable from clinical echo); feeding it to training is an
    *oracle upper bound*. Because even real IBFE ``f`` is not independent of the
    trajectory -- the structure moves with the interpolated fluid velocity
    (``dX/dt = u``), ``f`` closes the band momentum balance by construction, and
    its very support marks the instantaneous wall position -- the shuffle must not
    assume zero circularity. Modes:

    * ``None``        -- ``f`` unchanged (the oracle itself).
    * ``"shuffle"``   -- **support-preserving** temporal shuffle: permute the
      forcing vectors *within the band only* (interior zeros stay zero), so the
      support is preserved while the per-point trajectory correspondence is
      destroyed.
    * ``"band_mask"`` -- replace band forcing with a purely **geometric**
      placeholder (mean band magnitude x inward unit vector toward the cavity
      centroid), zero outside the band: it carries only *where the wall is*, no
      trajectory-specific force. The contrast ``shuffle`` vs ``band_mask`` vs the
      exact oracle separates a physics benefit from mere wall-position marking.
    """
    if mode is None:
        return forcing
    band = forcing_band_mask(forcing)
    nb = int(band.sum())
    if nb == 0:
        return forcing
    if mode == "shuffle":
        gp = torch.Generator().manual_seed(seed + 991)
        idx = torch.nonzero(band, as_tuple=False).squeeze(1)
        perm = idx[torch.randperm(nb, generator=gp)]
        out = forcing.clone()
        out[idx] = forcing[perm]
        return out
    if mode == "band_mask":
        out = torch.zeros_like(forcing)
        centroid = coords[:, :dim].mean(dim=0, keepdim=True)
        d = centroid - coords[:, :dim]
        d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)
        mag = float(forcing[band].norm(dim=1).mean())
        out[band] = mag * d[band]
        return out
    raise ValueError(f"unknown forcing_control '{mode}' (expected None/'shuffle'/'band_mask')")


def geometric_band_template(coords: torch.Tensor, dim: int,
                            band_frac: float = 0.35):
    """Geometric wall-band template for the deliverable Model B forcing ansatz.

    Returns ``(direction, weight)`` where ``direction`` ``(N, dim)`` is the inward
    unit vector toward the cavity centroid and ``weight`` ``(N, 1)`` ramps from 0 in
    the interior to 1 at the wall (over the outer ``band_frac`` of the normalised
    radius). Both are derived purely from the fluid point-cloud geometry (centroid +
    radial extent), i.e. from information available in wall kinematics -- **not** from
    the true forcing -- so feeding this template to :class:`ActivationForcing` leaks
    no trajectory information.
    """
    c = coords[:, :dim]
    centroid = c.mean(dim=0, keepdim=True)
    d = c - centroid
    r = d.norm(dim=1, keepdim=True)
    rmax = torch.quantile(r, 0.98).clamp_min(1e-9)
    rn = r / rmax
    inward = -d / r.clamp_min(1e-12)
    bf = max(float(band_frac), 1e-6)
    weight = ((rn - (1.0 - bf)) / bf).clamp(0.0, 1.0)
    return inward, weight


def _beam_dirs(X: torch.Tensor, transducer, dim: int) -> torch.Tensor:
    tr = torch.tensor(transducer, dtype=X.dtype, device=X.device).reshape(1, dim)
    d = X[:, :dim] - tr
    return d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)


def build_doppler_from_frames(frames, transducers: Sequence,
                              noise_level: float = 0.05, seed: int = 0):
    """Project ``velocity_fluid`` onto each window's beam -> single-component data.

    Returns ``(data_X, beam_dir, v_beam)`` stacked over all windows.
    """
    dim = frames.spatial_dim
    g = torch.Generator().manual_seed(seed)
    X = frames.coords_fluid
    vel = frames.velocity_fluid
    Xs, beams, vs = [], [], []
    for w in transducers:
        b = _beam_dirs(X, w, dim)
        vb = (vel * b).sum(dim=1, keepdim=True)
        if noise_level > 0:
            vb = vb + noise_level * vb.abs().mean() * torch.randn(
                vb.shape, generator=g, dtype=vb.dtype)
        Xs.append(X.clone()); beams.append(b); vs.append(vb)
    return torch.cat(Xs, 0), torch.cat(beams, 0), torch.cat(vs, 0)


def make_ibfe_batch_builder(frames, transducers: Optional[Sequence] = None,
                            noise_level: float = 0.05, seed: int = 0,
                            n_data: int = 3072, n_col: int = 3072,
                            n_wall: int = 768, n_valve: int = 256,
                            use_traction: bool = False,
                            predict_scalar: bool = False,
                            shuffle_forcing: bool = False,
                            forcing_control: Optional[str] = None,
                            param_forcing: bool = False,
                            band_frac: float = 0.35,
                            planes: Optional[Sequence] = None,
                            tracking: Optional[str] = None):
    """Return a ``build_batches(step)`` closure sourced from ``frames``.

    **Forcing is evaluation-only ground truth (an oracle), not a deliverable
    Model B input** -- it cannot be measured from clinical echo. Feeding it to
    training is an upper-bound experiment. ``shuffle_forcing`` / ``forcing_control``
    run the oracle forcing diagnostic via :func:`apply_forcing_control`:

    * ``shuffle_forcing=True`` (== ``forcing_control="shuffle"``) -- a
      **support-preserving** temporal shuffle (permute forcing *within the IB wall
      band*, interior zeros kept zero);
    * ``forcing_control="band_mask"`` -- a geometric wall-position placeholder.

    Real IBFE forcing is band-localized and not trajectory-independent (structure
    moves with ``u``; ``f`` closes the band momentum balance; its support marks the
    wall), so the exact / shuffle / band-mask contrast is what separates a physics
    benefit from wall-position marking -- do not assume zero circularity. Outside
    the band ``f = 0``, so Model B equals Model A there.

    ``planes`` (a list of :class:`~pinnecho.data.acquisition.ImagingPlane`) selects
    the realistic multi-plane acquisition geometry: each echo view only sees the
    fluid points inside its imaging-plane slab (apical / parasternal views). When
    given it supersedes ``transducers`` (whole-volume point windows). Build them
    with :func:`~pinnecho.data.acquisition.standard_views_from_frames`.

    ``tracking`` (a :mod:`~pinnecho.data.tracking_noise` preset name) optionally
    corrupts ``velocity_wall`` into a contour-tracking estimate for Model A.
    ``None`` / ``"exact"`` leave the FSI wall velocity untouched (the default,
    so Ablation 7/8 coincident-wall isolation is unchanged). Pass ``"ste"`` for
    the literature-calibrated tracking error when comparing a realistic Model A
    against an accurate-wall Model B.
    """
    from .ibfe_io import frames_to_dtype
    from .tracking_noise import apply_tracking_noise, resolve_tracking
    frames = frames_to_dtype(frames, torch.get_default_dtype())
    if tracking is None or resolve_tracking(tracking).name == "exact":
        wall_velocity = frames.velocity_wall
    else:
        wall_velocity = apply_tracking_noise(
            frames.velocity_wall, tracking, seed=seed)
    dim = frames.spatial_dim
    if planes is not None:
        from .acquisition import build_multiplane_doppler_from_frames
        data_X, beam_dir, v_beam = build_multiplane_doppler_from_frames(
            frames, planes, noise_level=noise_level, seed=seed)
    else:
        if transducers is None:
            transducers = default_windows_from_frames(frames, n_windows=2)
        data_X, beam_dir, v_beam = build_doppler_from_frames(
            frames, transducers, noise_level=noise_level, seed=seed)
    g = torch.Generator().manual_seed(seed)

    forcing_fluid = frames.forcing_fluid
    mode = forcing_control if forcing_control is not None else (
        "shuffle" if shuffle_forcing else None)
    if mode is not None and forcing_fluid.shape[0] > 1:
        forcing_fluid = apply_forcing_control(
            forcing_fluid, frames.coords_fluid, mode, seed, dim)

    has_valve = frames.coords_mitral.shape[0] > 0

    tmpl_dir = tmpl_w = None
    if param_forcing:
        tmpl_dir, tmpl_w = geometric_band_template(
            frames.coords_fluid, dim, band_frac=band_frac)

    def _idx(n_total, n):
        if n_total == 0:
            return torch.zeros(0, dtype=torch.long)
        if n >= n_total:
            return torch.arange(n_total)
        return torch.randint(0, n_total, (n,), generator=g)

    def build_batches(step: int) -> Dict[str, dict]:
        di = _idx(data_X.shape[0], n_data)
        ci = _idx(frames.coords_fluid.shape[0], n_col)
        wi = _idx(frames.coords_wall.shape[0], n_wall)
        batches = {
            "data": {"X": data_X[di].clone(), "beam_dir": beam_dir[di],
                     "v_beam": v_beam[di]},
            "collocation": {"X": frames.coords_fluid[ci].clone(),
                            "forcing": forcing_fluid[ci]},
            "wall": {"X": frames.coords_wall[wi].clone(),
                     "u_wall": wall_velocity[wi]},
        }
        if param_forcing:
            batches["collocation"]["forcing_template_dir"] = tmpl_dir[ci]
            batches["collocation"]["forcing_template_w"] = tmpl_w[ci]
        if use_traction:
            batches["traction"] = {
                "X": frames.coords_wall[wi].clone(),
                "normals": frames.normals_wall[wi],
                "structure_traction": frames.traction_wall[wi]}
        if has_valve:
            vi = _idx(frames.coords_mitral.shape[0], n_valve)
            batches["valve"] = {"X": frames.coords_mitral[vi].clone(),
                                "u_valve": frames.velocity_mitral[vi]}
            if predict_scalar:
                batches["scalar_inflow"] = {"X": frames.coords_mitral[vi].clone()}
        return batches

    return build_batches


def build_model_for_ibfe(frames, backbone: str = "fsi_informed",
                         predict_scalar: bool = False, use_traction: bool = False,
                         init_seed: int = 0, model_overrides: Optional[dict] = None,
                         n_harmonics: int = 2):
    """Build the shared :class:`PINNNet` + a backbone-appropriate loss for ``frames``.

    ``backbone`` is one of:

    * ``"baseline"``     -- Model A (kinematic wall, no forcing);
    * ``"fsi_informed"`` -- the **oracle** Model B (handed the true IB forcing);
    * ``"fsi_param"``    -- the **deliverable** Model B: accurate FSI wall velocity +
      a low-dim :class:`~pinnecho.models.activation_forcing.ActivationForcing`
      ansatz (``T_max`` + timing ``g(t)``) with a geometric band template; it never
      sees the true forcing.
    """
    from .ibfe_io import frames_to_dtype
    frames = frames_to_dtype(frames, torch.get_default_dtype())
    dim = frames.spatial_dim
    lows, highs = input_bounds_from_frames(frames)
    s = scales_from_frames(frames)
    m = dict(spatial_dim=dim, width=96, depth=5, activation="tanh",
             fourier_features=0, predict_scalar=predict_scalar)
    if model_overrides:
        m.update(model_overrides)
    torch.manual_seed(init_seed)
    model = PINNNet(input_lows=lows, input_highs=highs,
                    velocity_scale=s["velocity"], pressure_scale=s["pressure"], **m)

    cont_scale = s["length"] / max(s["velocity"], 1e-30)
    mom_scale = s["length"] / max(frames.rho * s["velocity"] ** 2, 1e-30)
    is_oracle = backbone == "fsi_informed"
    is_param = backbone == "fsi_param"
    is_fsi = is_oracle or is_param
    if is_param:
        from ..models.activation_forcing import ActivationForcing
        # Initialise T_max at a physical body-force-density scale rho U^2 / L.
        init_amp = frames.rho * s["velocity"] ** 2 / max(s["length"], 1e-9)
        model.activation_forcing = ActivationForcing(
            period=frames.cycle_period, n_harmonics=n_harmonics, init_amp=init_amp)
    forcing_mode = "fsi" if is_oracle else ("param" if is_param else None)
    weights = LossWeights(data=10.0, pde=1.0, scalar=1.0 if predict_scalar else 0.0,
                          bc=10.0, ic=0.0, periodic=0.0,
                          traction=5.0 if use_traction else 0.0)
    loss_fn = CompositeLoss(
        rho=frames.rho, mu=frames.mu, weights=weights,
        anneal=AnnealSchedule(enabled=True, pde_warmup_frac=0.3),
        forcing=forcing_mode,
        wall_mode="fsi" if is_fsi else "kinematic",
        use_traction=use_traction,
        continuity_scale=cont_scale, momentum_scale=mom_scale,
        traction_scale=mom_scale, scalar_scale=s["length"] / max(s["velocity"], 1e-30),
    )
    return model, loss_fn


@torch.no_grad()
def _vel_pred(model, X, dim):
    f = model.fields(X)
    return torch.cat([f[k] for k in ("u", "v", "w")[:dim]], dim=1)


def evaluate_ibfe(model, frames) -> Dict[str, float]:
    """Held-out velocity / pressure / vorticity error vs the fluid ground truth."""
    from .ibfe_io import frames_to_dtype
    frames = frames_to_dtype(frames, next(model.parameters()).dtype)
    dim = frames.spatial_dim
    X = frames.coords_fluid
    vel_true = frames.velocity_fluid
    vel_pred = _vel_pred(model, X.clone(), dim)
    out: Dict[str, float] = {}
    comps = ("u", "v", "w")[:dim]
    for i, name in enumerate(comps):
        num = torch.linalg.norm(vel_pred[:, i] - vel_true[:, i])
        den = torch.linalg.norm(vel_true[:, i]) + 1e-12
        out[f"vel_relL2_{name}"] = float(num / den)
    out["vel_relL2_speed"] = float(
        torch.linalg.norm(vel_pred - vel_true) / (torch.linalg.norm(vel_true) + 1e-12))

    with torch.no_grad():
        p_pred = model.fields(X.clone())["p"]
    p_pred = p_pred - p_pred.mean()
    p_true = frames.pressure_fluid - frames.pressure_fluid.mean()
    out["pressure_relL2"] = float(
        torch.linalg.norm(p_pred - p_true) / (torch.linalg.norm(p_true) + 1e-12))

    # Ground-truth vorticity is not stored in IBFEFrames; report the model's
    # vorticity magnitude as a smoke check that gradients are non-degenerate.
    if dim == 2:
        vort_pred = ev.vorticity_2d(model, X.clone()).detach()
    else:
        vort_pred = ev.vorticity_3d(model, X.clone()).detach()
    out["vorticity_absmax"] = float(vort_pred.abs().max())
    return out


def train_ibfe(frames, backbone: str = "fsi_informed", steps: int = 1500,
               lbfgs_iters: int = 150, lr: float = 2e-3, seed: int = 0,
               predict_scalar: bool = False, use_traction: bool = False,
               transducers: Optional[Sequence] = None, noise_level: float = 0.05,
               model_overrides: Optional[dict] = None, verbose: bool = True,
               shuffle_forcing: bool = False, forcing_control: Optional[str] = None,
               band_frac: float = 0.35, n_harmonics: int = 2,
               planes: Optional[Sequence] = None,
               tracking: Optional[str] = None):
    """Train a backbone on ``frames`` end-to-end and return ``(model, metrics)``.

    ``backbone="fsi_param"`` trains the **deliverable** Model B (accurate FSI wall
    velocity + a low-dim activation-forcing ansatz, never the true ``f``); ``band_frac``
    sets the geometric band template width and ``n_harmonics`` the timing ``g(t)``
    resolution. ``shuffle_forcing=True`` / ``forcing_control`` run the **oracle**
    forcing diagnostic (support-preserving shuffle or geometric band-mask control;
    see :func:`make_ibfe_batch_builder` / :func:`apply_forcing_control`). Pass
    ``planes`` (from :func:`~pinnecho.data.acquisition.standard_views_from_frames`)
    to use the realistic multi-plane echo acquisition geometry instead of
    whole-volume point windows.

    ``tracking`` is a wall-tracking preset (see
    :mod:`pinnecho.data.tracking_noise`). It is applied **only** to the
    ``baseline`` backbone (Model A), so a realistic A-vs-B comparison is
    ``train_ibfe(..., backbone="baseline", tracking="ste")`` vs
    ``train_ibfe(..., backbone="fsi_param")`` (exact FSI wall). Default ``None``
    keeps walls coincident (Ablation 7/8 isolation).
    """
    model, loss_fn = build_model_for_ibfe(
        frames, backbone=backbone, predict_scalar=predict_scalar,
        use_traction=use_traction, init_seed=seed, model_overrides=model_overrides,
        n_harmonics=n_harmonics)
    loss_fn.anneal.total_steps = steps
    wall_tracking = tracking if backbone == "baseline" else None
    build_batches = make_ibfe_batch_builder(
        frames, transducers=transducers, noise_level=noise_level, seed=seed,
        use_traction=use_traction, predict_scalar=predict_scalar,
        shuffle_forcing=shuffle_forcing, forcing_control=forcing_control,
        param_forcing=(backbone == "fsi_param"), band_frac=band_frac,
        planes=planes, tracking=wall_tracking)
    cfg = TrainConfig(steps=steps, lr=lr, lbfgs_iters=lbfgs_iters,
                      log_every=max(1, steps // 6), seed=seed)
    logger = None
    if verbose:
        def logger(step, row):
            print(f"[ibfe {step:5d}] " + " ".join(
                f"{k}={row[k]:.3e}" for k in ("total", "data", "pde", "bc") if k in row))
    train_model(model, loss_fn, build_batches, cfg, logger=logger)
    return model, evaluate_ibfe(model, frames)
