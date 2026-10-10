"""Band-localized forcing *oracle* dry-run on synthetic data.

The Stage-1 forcing ablations (``ablation.py``, Ablation 6) used a *full-cavity*
manufactured forcing derived from the true ``u`` and found the FSI gradient
"benefit" to be trajectory-specific leakage. Real IB forcing is different in two
ways that this module reproduces on synthetic data, so the exact analysis that
will run on real export (after the resolution gate) can be pre-registered and
validated now:

1. **Band-localized.** Real IB forcing is nonzero only in a ~3-cell wall shell and
   exactly zero in the cavity interior, so Model B equals Model A over most of the
   cavity. We emulate this with ``synthetic_ibfe_frames_3d(..., forcing_band_frac=)``
   (and the 2D variant), which zeros the manufactured forcing outside a wall band.
2. **Oracle-only.** The forcing is evaluation-only ground truth, never a
   deliverable Model B input (it is unavailable from clinical echo). Feeding it to
   training is an *upper bound*, and it still carries trajectory information, so we
   compare against two controls from :func:`pinnecho.data.ibfe_dataset.apply_forcing_control`:
   a **support-preserving** temporal shuffle (permute within the band, keep interior
   zero) and a geometric **band-mask** placeholder (wall-position marking only).

Conditions trained on one shared band-localized ground truth:

* ``A (baseline)``            -- kinematic wall, no forcing;
* ``oracle-exact``           -- FSI backbone handed the true band forcing;
* ``oracle-shuffle``         -- FSI backbone, support-preserving band shuffle;
* ``oracle-band_mask``       -- FSI backbone, geometric band-mask control.

A benefit that survives *both* controls is the part attributable to genuine
physics rather than trajectory information or support geometry -- and even then it
is an oracle bound, not evidence the deliverable Model B (no ``f``) would reproduce
it. Kept CPU-friendly; runtime scales with ``conditions x seeds``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..data.ibfe_dataset import (band_coverage_fraction, default_windows_from_frames,
                                  train_ibfe)
from ..data.load_ibfe_output import (synthetic_ibfe_frames,
                                      synthetic_ibfe_frames_3d)

# Held-out reconstruction metrics reported for every condition.
REPORT_KEYS = ("vel_relL2_speed", "vel_relL2_u", "vel_relL2_v", "vel_relL2_w",
               "pressure_relL2")

# Paired contrasts (name -> (condition_a, condition_b)); mean_diff = a - b on an
# error metric, so positive means b has the lower error (b improves on a).
CONTRASTS = {
    "exact_vs_baseline": ("A (baseline)", "oracle-exact"),
    "exact_vs_shuffle": ("oracle-shuffle", "oracle-exact"),
    "exact_vs_band_mask": ("oracle-band_mask", "oracle-exact"),
}

# Contrasts for the deliverable 3-way (A / deliverable fsi_param / oracle).
DELIVERABLE_CONTRASTS = {
    "deliverable_vs_baseline": ("A (baseline)", "B-deliverable (fsi_param)"),
    "oracle_vs_deliverable": ("B-deliverable (fsi_param)", "oracle-exact"),
}


@dataclass
class OracleCondition:
    """One training condition for the band-localized oracle contrast."""

    name: str
    backbone: str
    forcing_control: Optional[str] = None


def default_oracle_conditions() -> List[OracleCondition]:
    return [
        OracleCondition("A (baseline)", backbone="baseline"),
        OracleCondition("oracle-exact", backbone="fsi_informed"),
        OracleCondition("oracle-shuffle", backbone="fsi_informed",
                        forcing_control="shuffle"),
        OracleCondition("oracle-band_mask", backbone="fsi_informed",
                        forcing_control="band_mask"),
    ]


def deliverable_conditions() -> List[OracleCondition]:
    """The deliverable 3-way: Model A, the deliverable Model B (``fsi_param``:
    wall kinematics + low-dim activation ansatz, no true ``f``), and the oracle
    (true ``f``) as the achievable upper bound."""
    return [
        OracleCondition("A (baseline)", backbone="baseline"),
        OracleCondition("B-deliverable (fsi_param)", backbone="fsi_param"),
        OracleCondition("oracle-exact", backbone="fsi_informed"),
    ]


CONDITION_SETS = {
    "oracle": default_oracle_conditions,
    "deliverable": deliverable_conditions,
}
CONTRAST_SETS = {
    "oracle": CONTRASTS,
    "deliverable": DELIVERABLE_CONTRASTS,
}


def run_oracle_one(frames, cond: OracleCondition, steps: int, lbfgs_iters: int,
                   lr: float, seed: int, use_traction: bool, noise_level: float,
                   n_windows: int) -> Dict[str, float]:
    """Train one condition on the shared band-localized ``frames``."""
    transducers = default_windows_from_frames(frames, n_windows=n_windows)
    _, metrics = train_ibfe(
        frames, backbone=cond.backbone, steps=steps, lbfgs_iters=lbfgs_iters,
        lr=lr, seed=seed, use_traction=use_traction, noise_level=noise_level,
        transducers=transducers, forcing_control=cond.forcing_control,
        verbose=False)
    return metrics


def run_band_oracle_sweep(config=None, frames=None,
                          conditions: Optional[Sequence[OracleCondition]] = None,
                          contrasts: Optional[Dict] = None,
                          dim: int = 3, band_frac: float = 0.25,
                          steps: int = 1500, lbfgs_iters: int = 150, lr: float = 2e-3,
                          seeds: Sequence[int] = (0, 1, 2),
                          use_traction: bool = False, noise_level: float = 0.05,
                          n_fluid: int = 4000, n_wall: int = 900, n_frames: int = 8,
                          n_windows: int = 2, data_seed: int = 0,
                          verbose: bool = True) -> Dict:
    """Train every condition on one shared band-localized ground truth; aggregate.

    Returns ``{"band_frac", "band_coverage", "dim", "records": [...],
    "contrasts": {...}}`` where ``records`` holds per-condition mean/std metrics and
    ``contrasts`` holds the paired exact-vs-{baseline,shuffle,band_mask} statistics.
    """
    if frames is None:
        if config is None:
            from ..config import Config
            config = Config()
        gen = synthetic_ibfe_frames_3d if dim == 3 else synthetic_ibfe_frames
        frames = gen(config, n_fluid=n_fluid, n_wall=n_wall, n_frames=n_frames,
                     with_valve=False, forcing_band_frac=band_frac, seed=data_seed)
    coverage = band_coverage_fraction(frames.forcing_fluid)
    conditions = list(conditions) if conditions is not None else default_oracle_conditions()

    per_cond: Dict[str, List[Dict[str, float]]] = {}
    records: List[Dict] = []
    for cond in conditions:
        rows: List[Dict[str, float]] = []
        for s in seeds:
            m = run_oracle_one(frames, cond, steps, lbfgs_iters, lr, s,
                               use_traction, noise_level, n_windows)
            rows.append(m)
            if verbose:
                print(f"[{cond.name} seed={s}] " + " ".join(
                    f"{k}={m.get(k, float('nan')):.3f}" for k in REPORT_KEYS))
        per_cond[cond.name] = rows
        rec = {"name": cond.name, "backbone": cond.backbone,
               "forcing_control": cond.forcing_control}
        for k in REPORT_KEYS:
            vals = np.array([d[k] for d in rows if k in d], dtype=float)
            rec[f"{k}_mean"] = float(vals.mean()) if vals.size else float("nan")
            rec[f"{k}_std"] = float(vals.std()) if vals.size else float("nan")
        records.append(rec)

    from .ablation import _paired_stats
    contrast_defs = contrasts if contrasts is not None else CONTRASTS
    contrast_out: Dict[str, Dict] = {}
    for cname, (a_name, b_name) in contrast_defs.items():
        if a_name not in per_cond or b_name not in per_cond:
            continue
        entry: Dict[str, Dict[str, float]] = {}
        for metric in ("pressure_relL2", "vel_relL2_speed"):
            a = [d.get(metric, np.nan) for d in per_cond[a_name]]
            b = [d.get(metric, np.nan) for d in per_cond[b_name]]
            entry[metric] = _paired_stats(a, b, higher_better=True)
        contrast_out[cname] = entry

    return {"band_frac": float(band_frac), "band_coverage": float(coverage),
            "dim": int(dim), "n_windows": int(n_windows),
            "records": records, "contrasts": contrast_out}


def plot_band_oracle(result: Dict, out_path, metric: str = "pressure_relL2"):
    """Bar chart of ``metric`` (mean +/- std) across the oracle conditions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    records = result["records"]
    names = [r["name"] for r in records]
    means = [r[f"{metric}_mean"] for r in records]
    stds = [r[f"{metric}_std"] for r in records]
    def _color(r):
        if r["backbone"] == "baseline":
            return "#55a868"
        if r["backbone"] == "fsi_param":
            return "#dd8452"
        return "#4c72b0" if r["forcing_control"] is None else "#c44e52"

    colors = [_color(r) for r in records]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(1.7 * len(names) + 2, 5))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel(f"{metric} (relative L2, lower is better)")
    ax.set_title(f"Band-localized forcing "
                 f"(band covers {result['band_coverage']:.0%} of cavity; "
                 f"green=A, orange=deliverable B, blue=oracle, red=controls)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def band_oracle_main() -> None:  # pragma: no cover - CLI wiring
    import argparse
    import json
    from pathlib import Path
    from ..config import load_config, Config

    ap = argparse.ArgumentParser(
        description="Band-localized forcing oracle dry-run (A / exact / shuffle / "
                    "band-mask) on synthetic data.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--conditions", default="oracle", choices=sorted(CONDITION_SETS),
                    help="'oracle' (A/exact/shuffle/band_mask) or 'deliverable' "
                         "(A / fsi_param deliverable Model B / oracle upper bound).")
    ap.add_argument("--dim", type=int, default=3, choices=[2, 3])
    ap.add_argument("--band-frac", type=float, default=0.25,
                    help="wall-band thickness in normalised radius (0..1).")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lbfgs-iters", type=int, default=150)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--traction", action="store_true",
                    help="also inject exact traction continuity (off by default to "
                         "isolate the forcing term).")
    ap.add_argument("--noise-level", type=float, default=0.05)
    ap.add_argument("--n-fluid", type=int, default=4000)
    ap.add_argument("--n-windows", type=int, default=2)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--out", default="docs/results/band_oracle")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64 if args.dtype == "float64" else torch.float32)
    config = load_config(args.config) if args.config else Config()
    result = run_band_oracle_sweep(
        config, conditions=CONDITION_SETS[args.conditions](),
        contrasts=CONTRAST_SETS[args.conditions],
        dim=args.dim, band_frac=args.band_frac, steps=args.steps,
        lbfgs_iters=args.lbfgs_iters, seeds=tuple(args.seeds),
        use_traction=args.traction, noise_level=args.noise_level,
        n_fluid=args.n_fluid, n_windows=args.n_windows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "band_oracle.json", "w") as fh:
        json.dump(result, fh, indent=2)
    for metric in ("pressure_relL2", "vel_relL2_speed"):
        p = plot_band_oracle(result, out / f"band_oracle_{metric}.png", metric=metric)
        print(f"wrote {p}")

    print(f"\nband_frac={result['band_frac']} -> band covers "
          f"{result['band_coverage']:.1%} of the sampled cavity (dim={result['dim']})")
    print(f"\n{'condition':20s} {'speed':>8s} {'u':>8s} {'v':>8s} {'w':>8s} {'p':>8s}")
    for r in result["records"]:
        print(f"{r['name']:20s} {r['vel_relL2_speed_mean']:8.3f} "
              f"{r['vel_relL2_u_mean']:8.3f} {r['vel_relL2_v_mean']:8.3f} "
              f"{r['vel_relL2_w_mean']:8.3f} {r['pressure_relL2_mean']:8.3f}")
    print("\npaired contrasts (mean_diff = a - b on error; +ve => b lower error):")
    for cname, entry in result["contrasts"].items():
        for metric, st in entry.items():
            print(f"  {cname:20s} {metric:16s} "
                  f"Δ={st['mean_diff']:+.3f} t={st['t_stat']:.2f} "
                  f"wins={int(st['n_wins'])}/{int(st['n'])} "
                  f"sign_p1={st['sign_p_one_sided']:.3f}")


if __name__ == "__main__":  # pragma: no cover - CLI wiring
    band_oracle_main()
