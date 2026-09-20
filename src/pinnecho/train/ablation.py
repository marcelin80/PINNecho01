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

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

from .train import (
    TrainConfig,
    build_toy_dataset,
    build_toy_model,
    make_toy_batch_builder,
    train_model,
)


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
    parser.add_argument("--which", choices=["isolation", "perturbation", "both"],
                        default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--lbfgs-iters", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--levels", type=float, nargs="+",
                        default=[0.0, 0.05, 0.10, 0.20],
                        help="traction perturbation levels for the stress test")
    parser.add_argument("--out", default=None, help="write ablations.json here")
    parser.add_argument("--dtype", choices=["float32", "float64"],
                        default="float64")
    args = parser.parse_args()

    torch.set_default_dtype(
        torch.float64 if args.dtype == "float64" else torch.float32)
    config = load_config(args.config) if args.config else Config()

    payload: Dict[str, dict] = {"seeds": list(args.seeds), "steps": args.steps}

    if args.which in ("isolation", "both"):
        iso = run_physics_isolation(
            config, seeds=args.seeds, steps=args.steps,
            lbfgs_iters=args.lbfgs_iters, lr=args.lr)
        print("\n################ ABLATION 1: physics-term isolation "
              "(matched exact wall) ################")
        print(_fmt_table(iso))
        payload["isolation"] = {k: {m: list(v) for m, v in r.items()}
                                for k, r in iso.items()}

    if args.which in ("perturbation", "both"):
        pert = run_traction_perturbation(
            config, seeds=args.seeds, levels=args.levels, steps=args.steps,
            lbfgs_iters=args.lbfgs_iters, lr=args.lr)
        print("\n################ ABLATION 2: traction-uncertainty stress test "
              "################")
        print(_fmt_table(pert))
        payload["perturbation"] = {k: {m: list(v) for m, v in r.items()}
                                   for k, r in pert.items()}

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
