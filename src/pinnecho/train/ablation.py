"""Scientific-control ablations for the FSI-informed backbone claim.

The A-vs-B toy comparison in :mod:`pinnecho.train.train` is a *fair-architecture*
comparison, but two confounds make its headline pressure gain (corr 0.24 -> 0.996)
an upper bound rather than clean evidence that "FSI helps":

1. **Circularity of the traction target.** The synthetic ground truth is a
   *manufactured* incompressible solution: a prescribed divergence-free velocity
   field plus an analytic pressure ``p``, with the FSI forcing defined as the
   exact momentum residual. It is NOT produced by solving myocardial
   elastodynamics. The traction handed to Model B is therefore
   ``t = -p n + mu (grad u + grad u^T) n`` differentiated from that same analytic
   ``p``. Feeding it as a boundary constraint injects the pressure at the wall
   almost verbatim, so near-perfect pressure recovery is close to tautological.

2. **Asymmetric boundary information.** Model A was given a *noisy* contour
   tracking wall velocity while Model B got the exact wall velocity, so the
   comparison partly measured "clean vs noisy boundary info" rather than the
   physics term itself.

This module runs the two controls requested in review, entirely within the
synthetic setup and reusing the existing training infrastructure:

* :func:`run_physics_isolation` -- give **both** A and B the *exact* wall
  velocity, and separate the effect of (i) the FSI momentum forcing alone
  (``B_forcing``) from (ii) additionally constraining traction
  (``B_forcing_traction``). ``A_noisy`` is kept only as a reference for how much
  of the original gap was boundary noise. If ``B_forcing`` (forcing only, no
  traction) already beats ``A_exact``, that is the pure value of the physics
  term; if essentially all the pressure gain lives in ``B_forcing_traction``,
  that supports the circularity concern.

* :func:`run_traction_perturbation` -- a stress test: perturb the traction
  target that Model B receives with a systematic scale error (stiffness /
  contractility misestimate) plus elementwise noise at levels
  ``eps in {0, 0.05, 0.10, 0.20}`` and watch how fast the pressure recovery
  degrades. A real FSI solver only *estimates* traction, so if the 0.996 result
  collapses under a few percent perturbation it is an idealised ceiling.

Both drivers aggregate over seeds (mean +/- std) so the conclusions are not a
single-seed artefact.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .train import (
    TrainConfig,
    build_toy_dataset,
    build_toy_model,
    make_toy_batch_builder,
    train_model,
)

# Complementary acoustic windows used when a run requests >1 window (apical
# default + parasternal-like + a third oblique view), matching the observability
# sweep's convention so cross-beam observability improves with window count.
SECOND_WINDOW = (0.07, -0.02)
THIRD_WINDOW = (-0.07, -0.02)


def _apply_windows(config, n_windows: int):
    """Return a config copy whose Doppler acquisition uses ``n_windows`` views."""
    cfg = copy.deepcopy(config)
    d = cfg.doppler
    extra = [SECOND_WINDOW, THIRD_WINDOW]
    if n_windows <= 1:
        d.transducers = ()
    else:
        d.transducers = tuple([d.transducer] + extra[: n_windows - 1])
    return cfg


# Metrics we summarise (lower is better unless it ends in ``_corr``).
KEY_METRICS = (
    "pressure_relL2", "pressure_corr",
    "vorticity_relL2", "vorticity_corr",
    "wss_relL2", "wss_corr",
    "vel_relL2_u", "vel_relL2_v", "vel_relL2_speed",
)


@dataclass
class Variant:
    """One trainable configuration in an ablation."""

    name: str
    backbone: str            # "baseline" | "fsi_informed"
    wall_target: str         # "kinematic" | "fsi"
    use_traction: bool = False
    traction_key: str = "wall_traction"
    forcing_key: Optional[str] = None
    note: str = ""


def perturb_traction(t_true: torch.Tensor, level: float,
                     seed: int = 0) -> torch.Tensor:
    """Perturb an exact traction target to emulate FSI-estimation uncertainty.

    ``level`` (``eps``) is the relative uncertainty. We apply *both* a single
    systematic multiplicative scale error (a stiffness / active-contraction
    misestimate biases the whole stress field coherently) and independent
    elementwise Gaussian noise scaled by the field RMS::

        t_pert = t_true * (1 + eps * z_scale) + (eps * rms) * noise

    with ``z_scale ~ N(0,1)`` drawn once per call and ``noise ~ N(0,1)``
    elementwise. ``level == 0`` returns the exact traction unchanged.
    """
    if level <= 0.0:
        return t_true.clone()
    dtype = t_true.dtype
    g = torch.Generator().manual_seed(seed + 101)
    rms = float(t_true.pow(2).mean().sqrt().clamp_min(1e-12))
    z_scale = torch.randn((), generator=g, dtype=dtype)
    noise = torch.randn(t_true.shape, generator=g, dtype=dtype)
    return t_true * (1.0 + level * z_scale) + (level * rms) * noise


def _run_variant(config, dataset, variant: Variant, *, steps: int,
                 lbfgs_iters: int, lr: float, seed: int,
                 verbose: bool = False) -> Dict[str, float]:
    """Train one variant on the shared dataset and return held-out metrics."""
    from .evaluate_toy import evaluate_toy

    model, loss_fn = build_toy_model(
        config, dataset, backbone=variant.backbone, init_seed=seed,
        use_traction=variant.use_traction,
    )
    loss_fn.anneal.total_steps = steps
    build_batches = make_toy_batch_builder(
        dataset, seed=seed, with_traction=variant.use_traction,
        wall_target=variant.wall_target, traction_key=variant.traction_key,
        forcing_key=variant.forcing_key,
    )
    cfg = TrainConfig(steps=steps, lr=lr, lbfgs_iters=lbfgs_iters,
                      log_every=max(1, steps // 4), seed=seed)
    logger = None
    if verbose:
        def logger(step, row, _n=variant.name):
            print(f"[{_n[:16]:16s} {step:5d}] " + " ".join(
                f"{k}={row[k]:.3e}" for k in ("total", "data", "pde", "bc")
                if k in row))
    train_model(model, loss_fn, build_batches, cfg, logger=logger)
    return evaluate_toy(model, dataset_lv(dataset, config), config)


def dataset_lv(dataset, config):
    """Re-materialise the synthetic-LV object matching a toy dataset.

    ``build_toy_dataset`` returns ``(dataset, lv)``; when a caller only kept the
    dataset we rebuild the analytic LV from the same config (it is stateless /
    deterministic), so evaluation can query exact ground truth.
    """
    lv = dataset.extras.get("_lv")
    if lv is not None:
        return lv
    from ..data.synthetic_lv import SyntheticLVFSI
    return SyntheticLVFSI(config.geometry, config.flow,
                          dtype=torch.get_default_dtype())


def _aggregate(rows: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    """Mean/std over seed runs for each metric present in every row."""
    out: Dict[str, Tuple[float, float]] = {}
    if not rows:
        return out
    keys = set(rows[0])
    for r in rows[1:]:
        keys &= set(r)
    for k in sorted(keys):
        vals = torch.tensor([float(r[k]) for r in rows])
        out[k] = (float(vals.mean()), float(vals.std(unbiased=False)))
    return out


def run_physics_isolation(config, *, seeds: Sequence[int] = (0, 1),
                          steps: int = 2500, lbfgs_iters: int = 200,
                          lr: float = 2e-3, verbose: bool = True,
                          ) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Isolate the pure physics-term effect with matched boundary information.

    Returns a dict ``variant_name -> {metric: (mean, std)}`` aggregated over
    ``seeds``. All variants share one dataset per seed (built with the default
    tracking noise so both the exact and noisy wall velocities are available).
    """
    variants = [
        Variant("A_noisy", "baseline", "kinematic", note="reference: noisy wall"),
        Variant("A_exact", "baseline", "fsi", note="exact wall, no physics term"),
        Variant("B_forcing", "fsi_informed", "fsi",
                note="exact wall + FSI momentum forcing (no traction)"),
        Variant("B_forcing_traction", "fsi_informed", "fsi", use_traction=True,
                note="exact wall + forcing + exact traction constraint"),
    ]
    per_variant: Dict[str, List[Dict[str, float]]] = {v.name: [] for v in variants}
    for seed in seeds:
        dataset, lv = build_toy_dataset(config, seed=seed)
        dataset.extras["_lv"] = lv
        for v in variants:
            if verbose:
                print(f"\n=== [seed {seed}] variant {v.name}: {v.note} ===")
            m = _run_variant(config, dataset, v, steps=steps,
                             lbfgs_iters=lbfgs_iters, lr=lr, seed=seed,
                             verbose=verbose)
            per_variant[v.name].append(m)
    return {name: _aggregate(rows) for name, rows in per_variant.items()}


def run_traction_perturbation(config, *, seeds: Sequence[int] = (0, 1),
                              levels: Sequence[float] = (0.0, 0.05, 0.10, 0.20),
                              steps: int = 2500, lbfgs_iters: int = 200,
                              lr: float = 2e-3, verbose: bool = True,
                              ) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Stress-test Model B by perturbing its traction target.

    All runs use the exact wall velocity (fair) and the FSI forcing; only the
    traction-continuity target is perturbed by relative ``level``. Returns a dict
    keyed by ``"eps=<level>"`` -> ``{metric: (mean, std)}``.
    """
    per_level: Dict[str, List[Dict[str, float]]] = {}
    for seed in seeds:
        dataset, lv = build_toy_dataset(config, seed=seed)
        dataset.extras["_lv"] = lv
        t_true = dataset.extras["wall_traction"]
        for eps in levels:
            key = f"eps={eps:.2f}"
            tk = "wall_traction" if eps <= 0 else f"wall_traction_p{int(eps*100)}"
            if tk not in dataset.extras:
                dataset.extras[tk] = perturb_traction(t_true, eps, seed=seed)
            v = Variant(f"B_traction_{key}", "fsi_informed", "fsi",
                        use_traction=True, traction_key=tk,
                        note=f"traction perturbed by {eps:.0%}")
            if verbose:
                print(f"\n=== [seed {seed}] {v.name}: {v.note} ===")
            m = _run_variant(config, dataset, v, steps=steps,
                             lbfgs_iters=lbfgs_iters, lr=lr, seed=seed,
                             verbose=verbose)
            per_level.setdefault(key, []).append(m)
    return {key: _aggregate(rows) for key, rows in per_level.items()}


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson correlation between two 1-D sequences (nan-safe, 0 if degenerate)."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denom) if denom > 1e-12 else 0.0


def forcing_pressure_coupling(config, seed: int = 0, n: int = 4000) -> Dict[str, float]:
    """Quantify what the synthetic FSI forcing "leaks" -- pressure AND vorticity.

    The manufactured forcing is the *full* momentum residual
    ``f = rho Du/Dt + grad p - mu lap u``. It therefore explicitly contains:

    * the **pressure gradient** ``grad p`` (the pressure leak, Check 0), and
    * the **viscous term** ``-mu lap u``, which for an incompressible field equals
      ``mu curl(omega)`` (identity ``lap u = -curl(omega)``). So the forcing also
      carries the curl of vorticity -- a first-derivative-of-vorticity field that
      directly informs vorticity and wall shear stress.

    We decompose ``f`` into gradient / viscous(=vorticity-curl) / inertial parts
    and report, for each, the alignment (Pearson) with ``f`` and the magnitude
    share. The pressure-free residual ``g = f - grad p`` isolates the
    viscous+inertial content; ``corr(g, mu curl omega)`` is the vorticity analogue
    of Check 0's pressure gate.

    This is the **redesign gate**: a pressure-independent forcing must lower
    ``corr(f, grad p)``, but per the Helmholtz argument that pushes weight onto the
    solenoidal part -- which is exactly ``mu curl omega`` -- so the redesign must
    ALSO keep ``corr(f, mu curl omega)`` in check or it merely swaps a pressure
    circularity for a vorticity one. Hence two gates, not one.
    """
    from ..data.synthetic_lv import SyntheticLVFSI
    from ..physics import operators as ops

    dtype = torch.get_default_dtype()
    lv = SyntheticLVFSI(config.geometry, config.flow, dtype=dtype)
    mu = float(config.flow.viscosity)
    rng = np.random.default_rng(seed)
    t_values = np.linspace(0.0, config.flow.period, 24)
    X = lv.sample_interior(n, t_values, rng).to(dtype).requires_grad_(True)

    uv = lv._velocity_from_graph(X)
    u, v = uv[:, 0:1], uv[:, 1:2]
    gradp = ops.grad(lv.pressure(X), X)[:, :2]
    # Viscous term of the forcing, -mu lap u, and its identity form mu curl(omega).
    visc = -mu * torch.cat([ops.laplacian(u, X), ops.laplacian(v, X)], dim=1)
    omega = ops.curl_z(u, v, X)
    gomega = ops.grad(omega, X)
    mu_curl_omega = mu * torch.cat([gomega[:, 1:2], -gomega[:, 0:1]], dim=1)
    f = lv.forcing(X)
    inertial = f - gradp - visc  # = rho Du/Dt

    f = f.detach(); gradp = gradp.detach(); visc = visc.detach()
    mu_curl_omega = mu_curl_omega.detach(); inertial = inertial.detach()
    g = f - gradp
    nf = f.norm().clamp_min(1e-12)
    ng = g.norm().clamp_min(1e-12)
    return {
        # --- pressure leak (Check 0, original) ---
        "corr_forcing_gradp": _pearson(f.reshape(-1), gradp.reshape(-1)),
        "corr_inertialviscous_gradp": _pearson(g.reshape(-1), gradp.reshape(-1)),
        "gradp_magnitude_fraction": float(gradp.norm() / nf),
        # --- vorticity-curl leak (the new gate) ---
        "corr_forcing_muCurlOmega": _pearson(f.reshape(-1), mu_curl_omega.reshape(-1)),
        "corr_pressurefree_muCurlOmega": _pearson(g.reshape(-1), mu_curl_omega.reshape(-1)),
        "muCurlOmega_magnitude_fraction_f": float(mu_curl_omega.norm() / nf),
        "muCurlOmega_magnitude_fraction_g": float(mu_curl_omega.norm() / ng),
        # --- inertial (advective) part, the remaining leak ---
        "corr_forcing_inertial": _pearson(f.reshape(-1), inertial.reshape(-1)),
        "inertial_magnitude_fraction": float(inertial.norm() / nf),
        # --- identity sanity: -mu lap u ?= mu curl(omega) ---
        "identity_visc_curl_relerr": float(
            (visc - mu_curl_omega).norm() / mu_curl_omega.norm().clamp_min(1e-12)),
        "forcing_rms": float(f.pow(2).mean().sqrt()),
        "gradp_rms": float(gradp.pow(2).mean().sqrt()),
    }


def run_pressure_observability(config, *, seeds: Sequence[int] = (0, 1, 2),
                               windows: Sequence[int] = (1, 2, 3),
                               steps: int = 2500, lbfgs_iters: int = 200,
                               lr: float = 2e-3, verbose: bool = True,
                               ) -> Tuple[Dict[str, Dict[str, Tuple[float, float]]],
                                          Dict[str, float]]:
    """Is the ``A_exact`` pressure failure a *downstream symptom of observability*?

    Runs the **baseline** backbone (no forcing, no traction) with the **exact**
    wall velocity across increasing acoustic-window counts, so only the interior
    velocity observability changes. Returns ``(per_window, coupling)`` where
    ``per_window`` maps ``"windows=k" -> {metric: (mean, std)}`` and ``coupling``
    reports the across-run Pearson correlation between the interior cross-beam
    velocity error and the recovered pressure correlation. A strong (negative)
    coupling means the pressure failure is inherited from the velocity field, not
    from the missing physics term.
    """
    variant = Variant("A_exact", "baseline", "fsi",
                      note="baseline + exact wall, no physics term")
    per_window: Dict[str, List[Dict[str, float]]] = {}
    runs: List[Dict[str, float]] = []
    for w in windows:
        cfg = _apply_windows(config, w)
        key = f"windows={w}"
        for seed in seeds:
            dataset, lv = build_toy_dataset(cfg, seed=seed)
            dataset.extras["_lv"] = lv
            if verbose:
                print(f"\n=== [seed {seed}] {key}: {variant.note} ===")
            m = _run_variant(cfg, dataset, variant, steps=steps,
                             lbfgs_iters=lbfgs_iters, lr=lr, seed=seed,
                             verbose=verbose)
            m = dict(m, n_windows=float(w))
            per_window.setdefault(key, []).append(m)
            runs.append(m)
    coupling = {
        "corr_velU_pressureCorr": _pearson(
            [r["vel_relL2_u"] for r in runs], [r["pressure_corr"] for r in runs]),
        "corr_velSpeed_pressureCorr": _pearson(
            [r["vel_relL2_speed"] for r in runs], [r["pressure_corr"] for r in runs]),
        "corr_velU_pressureRelL2": _pearson(
            [r["vel_relL2_u"] for r in runs], [r["pressure_relL2"] for r in runs]),
        "n_runs": float(len(runs)),
    }
    return {k: _aggregate(v) for k, v in per_window.items()}, coupling


def run_observability_physics(config, *, seeds: Sequence[int] = (0, 1, 2),
                              windows: Sequence[int] = (1, 2),
                              steps: int = 2500, lbfgs_iters: int = 200,
                              lr: float = 2e-3, verbose: bool = True,
                              ) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Can FSI forcing substitute for an extra acoustic window? (velocity/gradients)

    Grid of ``windows x {baseline, fsi_forcing}`` with the **exact** wall for both
    and **no traction** (so pressure is never injected -- this comparison is
    non-circular and restricted to the velocity field and its gradients). The key
    contrast is ``windows=1 + forcing`` vs ``windows=2 + baseline``. Returns a dict
    keyed ``"w{n}_{baseline|forcing}" -> {metric: (mean, std)}``.
    """
    specs = [("baseline", "baseline", False), ("forcing", "fsi_informed", False)]
    per_variant: Dict[str, List[Dict[str, float]]] = {}
    for w in windows:
        cfg = _apply_windows(config, w)
        for tag, backbone, use_tr in specs:
            v = Variant(f"w{w}_{tag}", backbone, "fsi", use_traction=use_tr,
                        note=f"{w} window(s), {tag} (exact wall, no traction)")
            for seed in seeds:
                dataset, lv = build_toy_dataset(cfg, seed=seed)
                dataset.extras["_lv"] = lv
                if verbose:
                    print(f"\n=== [seed {seed}] {v.name}: {v.note} ===")
                m = _run_variant(cfg, dataset, v, steps=steps,
                                 lbfgs_iters=lbfgs_iters, lr=lr, seed=seed,
                                 verbose=verbose)
                per_variant.setdefault(v.name, []).append(m)
    return {k: _aggregate(v) for k, v in per_variant.items()}


def _paired_stats(a: Sequence[float], b: Sequence[float],
                  higher_better: bool = True) -> Dict[str, float]:
    """Per-seed paired comparison of ``a`` vs ``b`` (paired by index/seed).

    Returns the mean/std of the difference ``a - b``, a paired t-statistic
    ``mean / (std / sqrt(n))``, the count of pairs where ``a`` wins, and ``n``.
    No SciPy dependency: we report ``t`` and ``n`` so significance can be judged
    against a t-table (e.g. |t|>4.30 for a two-sided 0.05 test at dof=2).
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    n = int(min(x.size, y.size))
    x, y = x[:n], y[:n]
    diff = x - y
    mean = float(diff.mean()) if n else float("nan")
    std = float(diff.std(ddof=1)) if n > 1 else 0.0
    t = float(mean / (std / np.sqrt(n))) if (n > 1 and std > 1e-12) else float("nan")
    wins = int((diff > 0).sum()) if higher_better else int((diff < 0).sum())
    return {"mean_diff": mean, "std_diff": std, "t_stat": t,
            "n_wins": float(wins), "n": float(n)}


def run_forcing_perturbation(config, *, seeds: Sequence[int] = (0, 1, 2),
                             levels: Sequence[float] = (0.0, 0.10, 0.20, 0.30),
                             steps: int = 2500, lbfgs_iters: int = 200,
                             lr: float = 2e-3, verbose: bool = True,
                             ) -> Dict[str, object]:
    """Stress-test the *forcing* (vorticity/WSS analogue of Ablation 2).

    Ablation 4 used the **exact** forcing, so its gradient-quantity advantage is
    an idealised ceiling -- a real FSI solver only *estimates* the active-
    contraction forcing. Here we perturb the single-window ``fsi_forcing``
    (exact wall, no traction) collocation forcing by relative ``level`` and watch
    the vorticity/WSS correlation degrade. We also train the two-window baseline
    reference so we can report the **paired** ``w1_forcing(exact) - w2_baseline``
    difference for vorticity/WSS with per-seed statistics.

    Returns ``{"levels": {..}, "w2_baseline": {..}, "paired": {..}, "raw": {..}}``.
    """
    cfg1 = _apply_windows(config, 1)
    cfg2 = _apply_windows(config, 2)
    per_level: Dict[str, List[Dict[str, float]]] = {}
    base_rows: List[Dict[str, float]] = []
    raw: Dict[str, List[float]] = {}

    for seed in seeds:
        ds1, lv1 = build_toy_dataset(cfg1, seed=seed)
        ds1.extras["_lv"] = lv1
        f_true = ds1.col_forcing
        for eps in levels:
            key = f"eps={eps:.2f}"
            fk = "col_forcing" if eps <= 0 else f"col_forcing_p{int(eps*100)}"
            if eps > 0 and fk not in ds1.extras:
                ds1.extras[fk] = perturb_traction(f_true, eps, seed=seed)
            v = Variant(f"forcing_{key}", "fsi_informed", "fsi",
                        use_traction=False,
                        forcing_key=(None if eps <= 0 else fk),
                        note=f"1 window, forcing perturbed by {eps:.0%}")
            if verbose:
                print(f"\n=== [seed {seed}] {v.name}: {v.note} ===")
            m = _run_variant(cfg1, ds1, v, steps=steps, lbfgs_iters=lbfgs_iters,
                             lr=lr, seed=seed, verbose=verbose)
            per_level.setdefault(key, []).append(m)

        ds2, lv2 = build_toy_dataset(cfg2, seed=seed)
        ds2.extras["_lv"] = lv2
        vb = Variant("w2_baseline", "baseline", "fsi", note="2 windows, no FSI")
        if verbose:
            print(f"\n=== [seed {seed}] {vb.name}: {vb.note} ===")
        base_rows.append(_run_variant(cfg2, ds2, vb, steps=steps,
                                      lbfgs_iters=lbfgs_iters, lr=lr, seed=seed,
                                      verbose=verbose))

    # Paired stats: exact-forcing single window vs two-window baseline.
    exact_rows = per_level[f"eps={0.0:.2f}"]
    paired = {}
    for metric in ("vorticity_corr", "wss_corr"):
        raw[f"w1_forcing_{metric}"] = [r[metric] for r in exact_rows]
        raw[f"w2_baseline_{metric}"] = [r[metric] for r in base_rows]
        paired[metric] = _paired_stats(
            [r[metric] for r in exact_rows], [r[metric] for r in base_rows],
            higher_better=True)
    return {
        "levels": {k: _aggregate(v) for k, v in per_level.items()},
        "w2_baseline": _aggregate(base_rows),
        "paired": paired,
        "raw": raw,
    }


def _fmt_table(results: Dict[str, Dict[str, Tuple[float, float]]],
               metrics: Sequence[str] = KEY_METRICS) -> str:
    """Pretty ``mean+/-std`` table (rows = variants, cols = metrics)."""
    cols = [m for m in metrics if any(m in r for r in results.values())]
    header = f"{'variant':22s} " + " ".join(f"{m:>18s}" for m in cols)
    lines = [header, "-" * len(header)]
    for name, r in results.items():
        cells = []
        for m in cols:
            if m in r:
                mean, std = r[m]
                cells.append(f"{mean:8.4f}+/-{std:5.3f}")
            else:
                cells.append(f"{'-':>18s}")
        lines.append(f"{name:22s} " + " ".join(f"{c:>18s}" for c in cells))
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - CLI wiring
    """CLI: run the physics-isolation and/or traction-perturbation ablations."""
    import argparse
    import json
    from pathlib import Path
    from ..config import load_config, Config

    parser = argparse.ArgumentParser(
        description="FSI-backbone control ablations (synthetic data).")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--which",
        choices=["isolation", "perturbation", "coupling", "pressure-obs",
                 "substitution", "forcing-perturbation", "both", "all"],
        default="both",
        help="which experiment(s) to run ('both'=isolation+perturbation, "
             "'all'=every experiment)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--lbfgs-iters", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--levels", type=float, nargs="+",
                        default=[0.0, 0.05, 0.10, 0.20],
                        help="traction perturbation levels for the stress test")
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 2, 3],
                        help="acoustic-window counts for the observability runs")
    parser.add_argument("--out", default=None, help="write ablations.json here")
    parser.add_argument("--dtype", choices=["float32", "float64"],
                        default="float64")
    args = parser.parse_args()

    torch.set_default_dtype(
        torch.float64 if args.dtype == "float64" else torch.float32)
    config = load_config(args.config) if args.config else Config()

    def want(name: str) -> bool:
        if args.which == "all":
            return True
        if args.which == "both":
            return name in ("isolation", "perturbation")
        return args.which == name

    payload: Dict[str, dict] = {"seeds": list(args.seeds), "steps": args.steps}

    # A cheap analytic check (no training): how circular is the current forcing?
    if want("coupling") or args.which == "all":
        coup = {s: forcing_pressure_coupling(config, seed=s) for s in args.seeds}
        agg = _aggregate(list(coup.values()))
        print("\n################ CHECK 0: forcing <-> pressure-gradient coupling "
              "(circularity gate) ################")
        for k, (mean, std) in agg.items():
            print(f"  {k:32s} {mean:8.4f} +/- {std:5.3f}")
        payload["coupling"] = {k: list(v) for k, v in agg.items()}

    if want("isolation"):
        iso = run_physics_isolation(
            config, seeds=args.seeds, steps=args.steps,
            lbfgs_iters=args.lbfgs_iters, lr=args.lr)
        print("\n################ ABLATION 1: physics-term isolation "
              "(matched exact wall) ################")
        print(_fmt_table(iso))
        payload["isolation"] = {k: {m: list(v) for m, v in r.items()}
                                for k, r in iso.items()}

    if want("perturbation"):
        pert = run_traction_perturbation(
            config, seeds=args.seeds, levels=args.levels, steps=args.steps,
            lbfgs_iters=args.lbfgs_iters, lr=args.lr)
        print("\n################ ABLATION 2: traction-uncertainty stress test "
              "################")
        print(_fmt_table(pert))
        payload["perturbation"] = {k: {m: list(v) for m, v in r.items()}
                                   for k, r in pert.items()}

    if want("pressure-obs"):
        pobs, coupling = run_pressure_observability(
            config, seeds=args.seeds, windows=args.windows, steps=args.steps,
            lbfgs_iters=args.lbfgs_iters, lr=args.lr)
        print("\n################ ABLATION 3: pressure recovery vs velocity "
              "observability (baseline, exact wall) ################")
        print(_fmt_table(pobs))
        print("\n  across-run coupling (Pearson):")
        for k, v in coupling.items():
            print(f"    {k:28s} {v:8.4f}")
        payload["pressure_obs"] = {
            "per_window": {k: {m: list(v) for m, v in r.items()}
                           for k, r in pobs.items()},
            "coupling": coupling,
        }

    if want("substitution"):
        sub = run_observability_physics(
            config, seeds=args.seeds, windows=[w for w in args.windows if w <= 2],
            steps=args.steps, lbfgs_iters=args.lbfgs_iters, lr=args.lr)
        print("\n################ ABLATION 4: can FSI forcing substitute for an "
              "acoustic window? (velocity/gradients, non-circular) ################")
        print(_fmt_table(sub))
        payload["substitution"] = {k: {m: list(v) for m, v in r.items()}
                                   for k, r in sub.items()}

    if want("forcing-perturbation"):
        fp = run_forcing_perturbation(
            config, seeds=args.seeds, steps=args.steps,
            lbfgs_iters=args.lbfgs_iters, lr=args.lr)
        print("\n################ ABLATION 5: forcing-uncertainty stress test + "
              "paired stats (vorticity/WSS) ################")
        tbl = dict(fp["levels"])
        tbl["w2_baseline (ref)"] = fp["w2_baseline"]
        print(_fmt_table(tbl))
        print("\n  paired: w1_forcing(exact) - w2_baseline (higher corr = forcing wins)")
        for metric, st in fp["paired"].items():
            print(f"    {metric:16s} mean_diff={st['mean_diff']:+.4f} "
                  f"std={st['std_diff']:.4f} t={st['t_stat']:+.3f} "
                  f"wins={int(st['n_wins'])}/{int(st['n'])}")
        payload["forcing_perturbation"] = {
            "levels": {k: {m: list(v) for m, v in r.items()}
                       for k, r in fp["levels"].items()},
            "w2_baseline": {m: list(v) for m, v in fp["w2_baseline"].items()},
            "paired": fp["paired"],
            "raw": fp["raw"],
        }

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
