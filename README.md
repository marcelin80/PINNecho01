# PINNecho — FSI-Informed PINN for Intraventricular Doppler Flow Reconstruction

**Stage 1 · Synthetic data**

PINNecho is a physics-informed neural network (PINN) that reconstructs the full
2D intraventricular flow field — velocity, pressure, vorticity and blood
residence time — inside the left ventricle (LV) from **sparse, single-component,
Doppler-like** velocity measurements.

Its purpose is to compare two physics backbones and answer one scientific
question:

> Does informing the Navier–Stokes backbone with a cardiac **fluid–structure
> interaction (FSI)** simulation improve recovery of velocity-**gradient**
> quantities (vorticity, wall shear stress) and pressure, compared to the
> generic incompressible Navier–Stokes + kinematic no-slip approach used in the
> existing literature (AI-VFM, iVFM-PINN, CSF-PINN)?

This directly targets a weakness that CSF-PINN (Wong et al. 2025) reported in
their own paper: poor recovery of vorticity and wall shear stress (WSS).

> **Stage 1 scope.** No real patient data or clinical echo access is available
> yet. Everything here runs on **synthetic** data that mimics the output of an
> existing IBAMR/IBFE cardiac FSI pipeline. The synthetic ground truth is
> analytic, divergence-free, satisfies exact no-slip on a moving wall, and has a
> nontrivial pressure field — see [Synthetic ground truth](#synthetic-ground-truth).

---

## The two backbones

Both backbones enforce the **same** incompressible Navier–Stokes momentum
balance,

```
rho (u_t + (u · grad) u) = -grad p + mu * lap u + f
```

and differ only in the body-forcing term `f`:

| | Backbone A — `baseline` | Backbone B — `fsi_informed` |
|---|---|---|
| Momentum forcing `f` | `0` (generic incompressible NS) | FSI body forcing from the IBAMR/IBFE run |
| Wall BC | kinematic no-slip (velocity from moving contour) | kinematic no-slip (identical in Stage 1) |
| Mirrors | AI-VFM, iVFM-PINN, CSF-PINN | proposed method |

The FSI forcing `f` represents the **net effect of myocardial active contraction
and structural coupling** on the fluid — physics that a purely kinematic wall
boundary condition cannot capture. In Stage 1 the two wall velocities coincide
(the synthetic solution has exact no-slip), so the comparison **isolates the
effect of the FSI forcing** in the momentum equation. The wall boundary
condition lives in its own module (`physics/boundary.py`) so a later stage can
diverge the two wall velocities without touching the trainer.

---

## Synthetic ground truth

Because no real IBAMR/IBFE run is available in Stage 1, `data/synthetic_lv.py`
*manufactures* a self-consistent ground-truth flow field on a deforming
elliptical LV cavity:

* **Geometry** — the long/short semi-axes pulsate over the cardiac cycle while
  the 2D cavity **area is held constant**, so the wall motion is genuinely
  divergence-free (2D incompressibility) with no phantom sources.
* **Velocity** `u = u_wall + u_vortex`
  * `u_wall` — the affine, divergence-free wall-following velocity of the
    deforming ellipse (equals the true endocardial velocity at the wall).
  * `u_vortex = curl(psi)` with a stream function `psi = (1 - rho^2)^2 · (…)`
    that vanishes together with its gradient on the wall (`rho` = normalised
    radius, `rho = 1` on the wall). Hence the total field satisfies **exact
    no-slip**: `u = u_wall` on the endocardium.
* **Pressure** — a smooth manufactured field `p` (defined up to a constant).
* **FSI body forcing** `f` — defined as *exactly* the Navier–Stokes momentum
  residual of the manufactured `(u, p)`. By construction `(u, p, f)` satisfy
  incompressible NS **to machine precision** (verified in the test suite).

Every derivative (velocity from the stream function, vorticity, forcing) is
taken with `torch.autograd`, so there is no hand-derived algebra to get wrong.

Key properties (asserted in `tests/test_synthetic_lv.py`):

* `max |div u|`  ≈ 1e-15 (divergence-free)
* `max |u - u_wall|` at the wall ≈ 1e-16 (exact no-slip)
* momentum residual with the correct `f` ≈ 0 (NS-exact)

### Virtual Doppler acquisition

`data/doppler.py` emulates a color/PW-Doppler acquisition: for each sample point
it records only the velocity component **along the beam** (the direction from a
virtual transducer to the point), sparsely sampled per frame and corrupted with
Gaussian noise. This one-component, sparse, noisy signal is the **only** velocity
information the PINN sees — everything else (the other velocity component,
pressure, gradients) must be inferred through the physics.

---

## Why the FSI backbone should win

Given the sparse single-component data, the reconstruction is under-determined
and the physics residual selects the solution. Because the true field requires
the forcing `f`:

* **Baseline** enforces momentum with `f = 0`, which is inconsistent with the
  true `(u, p)`. In particular its pressure gradient is wrong by exactly the
  missing `f`, and the inconsistent momentum residual biases the velocity
  gradients — degrading pressure, vorticity and WSS.
* **FSI-informed** enforces the *same* momentum equation the ground truth
  satisfies, so data + consistent physics + no-slip pin the solution to the
  truth and gradient quantities are recovered far more faithfully.

---

## Repository structure

```
PINNecho01/
├── configs/                     # experiment configs (YAML)
│   ├── baseline.yaml            #   backbone A
│   ├── fsi_informed.yaml        #   backbone B
│   └── smoke.yaml               #   tiny/fast config for CI & demos
├── src/pinnecho/
│   ├── config.py                # dataclass config + YAML (de)serialisation
│   ├── cli.py                   # generate / train / evaluate / compare entry points
│   ├── pipeline.py              # reusable high-level orchestration
│   ├── data/
│   │   ├── synthetic_lv.py      # synthetic IBAMR/IBFE-like LV FSI ground truth
│   │   ├── doppler.py           # single-component sparse Doppler sampling
│   │   └── dataset.py           # measurement + collocation + wall point sets
│   ├── models/
│   │   ├── mlp.py               # Fourier-feature MLP
│   │   └── pinn.py              # (x,y,t) -> (u,v,p), with non-dimensionalisation
│   ├── physics/
│   │   ├── operators.py         # autograd grad / div / curl / laplacian / D/Dt
│   │   ├── navier_stokes.py     # incompressible NS residual (with forcing hook)
│   │   └── boundary.py          # no-slip wall BC residual
│   ├── train/
│   │   ├── losses.py            # data + continuity + momentum + wall losses
│   │   └── trainer.py           # Adam training loop, per-backbone forcing
│   ├── eval/
│   │   ├── metrics.py           # velocity / pressure / vorticity / WSS errors
│   │   ├── residence_time.py    # Lagrangian blood residence time
│   │   └── plots.py             # field / loss / comparison figures
│   └── utils/                   # seeding, logging
├── scripts/                     # thin CLI wrappers (run without install)
├── tests/                       # pytest suite (fast, small-scale)
├── pyproject.toml
└── requirements.txt
```

---

## Installation

CPU-only is fine (the defaults are tuned to run on a laptop):

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU wheel
pip install -e .            # installs pinnecho + console scripts
# or, without installing:
pip install -r requirements.txt
```

---

## Usage

The scripts under `scripts/` work directly from a checkout (they add `src/` to
the path); the equivalent console scripts (`pinnecho-*`) are available after
`pip install -e .`.

### Quick smoke run (seconds)

```bash
python scripts/compare_backbones.py --config configs/smoke.yaml --out outputs/smoke --iterations 300
```

### Generate a dataset

```bash
python scripts/generate_data.py --config configs/baseline.yaml --out data/generated/dataset.npz
```

### Train a single backbone

```bash
python scripts/train.py --config configs/fsi_informed.yaml --out outputs/fsi
python scripts/train.py --config configs/baseline.yaml     --out outputs/baseline
```

### Evaluate a checkpoint (with field plots)

```bash
python scripts/evaluate.py --checkpoint outputs/fsi/model.pt --out outputs/fsi/eval --plot
```

### Compare both backbones on identical data (the headline experiment)

```bash
python scripts/compare_backbones.py --config configs/baseline.yaml --out outputs/compare
```

This trains **both** backbones from the same initialisation on the **same**
dataset, evaluates every metric, prints a comparison table, and writes
`comparison.json`, `comparison.txt`, `loss_curves.png` and
`metric_comparison.png`.

---

## Metrics

All errors are relative L2 (lower is better), computed against the analytic
ground truth over interior points across the cardiac cycle:

| metric | quantity | derivative order |
|---|---|---|
| `rel_l2_speed` | velocity magnitude | 0 |
| `rel_l2_pressure` | pressure (mean-removed) | — |
| `rel_l2_vorticity` | out-of-plane vorticity `dv/dx − du/dy` | 1st |
| `rel_l2_wss` | wall shear stress vector on the endocardium | 1st |
| `rel_l2_residence_time` | Lagrangian blood residence-time map | integral |

The gradient-order and integral quantities (vorticity, WSS, residence time) are
exactly the ones the baseline literature struggles with and that the FSI-informed
backbone is designed to improve.

---

## Configuration

Every experiment is fully described by a single YAML file mapped onto the
dataclasses in `config.py`. Anything omitted falls back to documented defaults.
CLI flags `--backbone`, `--iterations`, `--seed` override the file. To sweep how
strongly the FSI forcing is trusted, set `physics.forcing_scale` between `0.0`
(collapses to baseline) and `1.0` (full FSI forcing).

---

## Tests

```bash
pytest
```

The suite covers the autograd operators (vs. closed-form derivatives), the
self-consistency of the synthetic FSI field (divergence-free, exact no-slip,
NS-exact forcing), Doppler projection/noise, config round-trips, and an
end-to-end training + evaluation smoke test.

---

## Roadmap beyond Stage 1

* Replace the manufactured field with a real IBAMR/IBFE cardiac FSI export.
* Diverge the baseline (kinematic) and FSI wall velocities (structural normal
  traction / slip).
* Extend to 3D (the coordinate/operator layout is already dimension-oriented).
* Move from a virtual transducer to realistic beam geometries and clutter/aliasing.

## References

* Wong et al. (2025), *CSF-PINN* — reports poor vorticity / WSS recovery.
* AI-VFM, iVFM-PINN — kinematic no-slip vector flow mapping baselines.
* Tancik et al. (2020), *Fourier Features Let Networks Learn High Frequency
  Functions in Low Dimensional Domains*.
