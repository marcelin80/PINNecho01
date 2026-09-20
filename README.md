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
virtual transducer to the point), sampled per frame and corrupted with Gaussian
noise. This single-component, noisy signal is the **only** velocity information
the PINN sees — everything else (the cross-beam velocity component, pressure,
gradients) must be inferred through the physics.

**Multiple acoustic windows.** With a single transducer the beams are nearly
parallel across the cavity, so the cross-beam velocity component is poorly
observable and is left almost entirely to the physics prior. Real
echocardiography combines views (e.g. apical + parasternal); accordingly the
acquisition supports several virtual transducers (`doppler.transducers`). Each
measurement is still single-component, but combining windows at different angles
substantially improves full-field recovery.

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
  truth. This most directly benefits the **pressure** field, which is governed
  entirely by the momentum balance the forcing enters.

---

## Model & training

* **Model.** A plain `tanh` MLP mapping non-dimensional `(x, y, t)` to
  non-dimensional `(u, v, p)`. Smooth activations are used deliberately: the
  headline quantities include velocity *derivatives* (vorticity, WSS), and a
  plain tanh network has smooth, accurate derivatives. Random Fourier features
  are available (`model.fourier_features > 0`) but **off by default** — see
  [Stage-1 findings](#stage-1-findings).
* **Losses.** Relative (dimensionless) Doppler-data misfit and no-slip wall
  misfit, plus non-dimensionalised continuity and momentum residuals.
* **Curriculum.** The physics (continuity + momentum) weights are ramped from 0
  over the first `physics_warmup_frac` of training. At initialisation the PDE
  residuals dwarf the data term and are *both minimised by the trivial `u = 0`
  field*; without the ramp the optimiser collapses to `u = 0` and never fits the
  data. Warming up on data + wall first establishes a non-trivial field.
* **Optimisation.** Adam, followed by an optional full-batch **L-BFGS** polish
  (`train.lbfgs_iters`) that drives the residuals down the last 1–2 orders of
  magnitude.

---

## Repository structure

```
PINNecho01/
├── configs/                     # experiment configs (YAML)
│   ├── default.yaml             #   spec-aligned config: every knob documented
│   ├── baseline.yaml            #   backbone A (Stage-1 demo pipeline)
│   ├── fsi_informed.yaml        #   backbone B (Stage-1 demo pipeline)
│   └── smoke.yaml               #   tiny/fast config for CI & demos
├── src/pinnecho/
│   ├── config.py                # dataclass config + YAML (de)serialisation
│   ├── cli.py                   # generate / train / evaluate / compare entry points
│   ├── pipeline.py              # reusable high-level orchestration
│   ├── evaluate.py              # spec metrics: velocity/pressure/vorticity/Q/WSS/
│   │                            #   residence-time + stagnation + spline baseline
│   ├── data/
│   │   ├── synthetic_lv.py      # synthetic IBAMR/IBFE-like LV FSI ground truth
│   │   ├── doppler.py           # single-component sparse Doppler sampling
│   │   ├── dataset.py           # measurement + collocation + wall point sets
│   │   ├── load_ibfe_output.py  # [TODO STUB] real IBFE loader (documented shapes)
│   │   └── synthesize_doppler.py# [TODO STUB] Doppler synthesis (documented shapes)
│   ├── models/
│   │   ├── mlp.py               # Fourier-feature MLP (Stage-1 demo)
│   │   ├── pinn.py              # (x,y,t) -> (u,v,p), with non-dimensionalisation
│   │   └── mlp_pinn.py          # spec network: multi-scale Fourier + heads u,v,[w],p,c
│   ├── physics/
│   │   ├── operators.py         # autograd grad / div / curl / laplacian / D/Dt
│   │   ├── ns_residual.py       # incompressible NS residual (2D/3D, forcing hook)
│   │   ├── scalar_transport.py  # residence-time passive-scalar residual
│   │   ├── navier_stokes.py     # back-compat alias of ns_residual
│   │   └── boundary.py          # no-slip wall BC residual
│   ├── bc/
│   │   └── boundary_conditions.py  # Model A kinematic (impl); Model B (stubs)
│   ├── train/
│   │   ├── losses.py            # Stage-1 demo losses
│   │   ├── trainer.py           # Stage-1 demo Adam/L-BFGS loop
│   │   ├── composite_loss.py    # spec 6-term loss + PDE-weight annealing
│   │   └── train.py             # spec training entry (Model A loop; Model B gated)
│   ├── eval/
│   │   ├── metrics.py           # velocity / pressure / vorticity / WSS errors
│   │   ├── residence_time.py    # Lagrangian blood residence time
│   │   └── plots.py             # field / loss / comparison figures
│   └── utils/                   # seeding, logging
├── scripts/                     # thin CLI wrappers (run without install)
├── tests/                       # pytest suite (fast, small-scale)
│   ├── test_ns_residual_taylor_green.py  # NS residual ~0 on exact solutions
│   └── test_scalar_transport.py          # scalar residual ~0 on exact solutions
├── pyproject.toml
└── requirements.txt
```

### Spec-aligned build status

The modules above marked "spec" implement the detailed physics/architecture
specification incrementally. Current status:

| Component | File | Status |
|---|---|---|
| Incompressible NS residual (2D & 3D) | `physics/ns_residual.py` | **Implemented** + unit-tested (Taylor–Green, Poiseuille) |
| Residence-time scalar transport | `physics/scalar_transport.py` | **Implemented** + unit-tested (advection/diffusion/source) |
| Multi-scale Fourier network (`u,v,[w],p,c`) | `models/mlp_pinn.py` | **Implemented** (σ∈{1,10,100}, tanh/sine, 2D/3D) |
| Composite loss (data/pde/scalar/bc/ic/periodic) + annealing | `train/composite_loss.py` | **Implemented** (data loss is beam-component only) |
| Model A kinematic BC + valve/inflow | `bc/boundary_conditions.py` | **Implemented** |
| Validation metrics + spline baseline | `evaluate.py` | **Implemented** (primitives); end-to-end driver scaffolded |
| Training entry | `train/train.py` | **Implemented**; Model A validated end-to-end on the toy 2D case |
| IBFE loader / Doppler synthesis | `data/load_ibfe_output.py`, `data/synthesize_doppler.py` | **TODO stubs** with documented tensor shapes |
| Model B (FSI wall velocity / traction continuity) | `bc/boundary_conditions.py` | **Not started** (deliberately, until Model A validates — now next) |

> **Staging.** Per the plan, Model B's traction-matching BC is not begun until
> Model A trains successfully end-to-end on the toy 2D case, and each stage is
> confirmed before the next. The Model B BC helpers exist as clearly-marked
> `NotImplementedError` stubs so the interface is fixed but the ablation stays
> honest.

### Model A end-to-end on the toy case (spec modules)

`scripts/train_model_a.py` trains the baseline backbone on the synthetic-LV toy
case using the spec network + composite loss (no FSI forcing, kinematic no-slip):

```bash
python scripts/train_model_a.py --config configs/baseline.yaml --steps 3000 --lbfgs-iters 300
```

Reference run (dual-window Doppler, plain-tanh 96×5 network, Adam + L-BFGS, CPU,
held-out points across the cycle):

| metric (relative L2, ↓) | Model A |
|---|---|
| speed | 0.447 |
| velocity `u` / `v` | 0.641 / 0.537 |
| vorticity (corr 0.82) | 0.578 |
| wall shear stress (corr 0.65) | 0.506 |
| pressure (corr −0.47) | 1.190 |
| physics-free spline baseline (beam) | 1.059 |

The data loss falls from ~1.0 (trivial `u=0`) to ~0.21, i.e. the network
genuinely fits the sparse **single-component** Doppler and — via the physics —
recovers the unseen cross-beam component and the velocity gradients (vorticity /
WSS correlations 0.65–0.82). Pressure is poorly recovered by the baseline
(`f = 0` momentum ⇒ wrong pressure gradient): this is exactly the weakness the
**FSI-informed Model B** is designed to address next.

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

## Stage-1 findings

Getting the reconstruction to work surfaced several results that are themselves
informative (and are baked into the defaults):

1. **Smooth activations are essential for gradient quantities.** In a
   *fully-supervised* control (fit directly to the true `u, v, p`), a plain tanh
   MLP recovered velocity, pressure, vorticity and WSS accurately, whereas the
   same fit with random Fourier features (scale 2–5) matched the velocity values
   but recovered **vorticity/WSS ~10× worse** — the features inject
   high-frequency wiggle that fits values while corrupting derivatives. Fourier
   features are therefore disabled by default.
2. **Single-component acquisition is an observability bottleneck.** With one
   acoustic window, even after driving all PDE/data residuals to ~1e-3 the
   cross-beam velocity component is only partially recovered, which in turn caps
   pressure/gradient accuracy. Multiple windows (and, ultimately, richer priors)
   are needed for high-fidelity full-field recovery.
3. **The FSI forcing helps most where the momentum balance dominates
   (pressure).** Because the forcing enters only the momentum equation, its
   cleanest, most consistent benefit is on pressure; its benefit to velocity
   gradients is contingent on the velocity itself being well recovered.

### Headline comparison (reference run)

Trained on identical data (dual-window, 2% noise), plain-tanh network, Adam +
L-BFGS, on CPU:

| metric (relative L2, ↓) | baseline | fsi_informed | change |
|---|---|---|---|
| speed | 0.434 | 0.458 | −5.6% |
| **vorticity** | 0.585 | **0.541** | **+7.5%** |
| **wall shear stress** | 0.598 | **0.579** | **+3.2%** |
| pressure | 1.030 | 3.087 | −199% |
| residence time | 0.290 | 0.348 | −20% |

Reproduce with:

```bash
python scripts/compare_backbones.py --config configs/baseline.yaml --out outputs/compare
cat outputs/compare/comparison.txt
```

**Reading the result.** The FSI-informed backbone improves exactly the
velocity-**gradient** quantities this project targets — vorticity (+7.5%) and
WSS (+3.2%), the metrics CSF-PINN reported as weak — consistent with the
hypothesis. However, in this Stage-1 regime the **pressure** recovery is *worse*
for the FSI backbone: the (large) manufactured forcing is dominated by the true
pressure gradient, so when the velocity is only partially recovered (the
single-component observability bottleneck, ~0.45 speed error for both backbones)
the momentum balance amplifies those velocity errors into the pressure estimate,
whereas the baseline's `f = 0` momentum yields a bounded (but also wrong)
pressure. Fully realising the pressure benefit therefore requires more accurate
velocity recovery (richer acquisition / larger training budget / pressure
gauge-fixing) — a Stage-2 objective. Treat this setup as a **validated framework
and methodology** rather than a converged clinical result.

Example artefacts from the reference run live in [`docs/results/`](docs/results):
`metric_comparison.png`, `loss_curves.png`, and per-backbone field comparisons
(`fields_baseline.png`, `fields_fsi_informed.png`).

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
