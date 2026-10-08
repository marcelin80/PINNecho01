# Observability of single-component Doppler flow reconstruction

**Standalone result — independent of the FSI / circularity discussion.**

This finding does *not* depend on the FSI backbone, on any forcing/traction term,
or on the manufactured-solution circularity that limits the pressure/gradient
claims (see [`ABLATIONS.md`](ABLATIONS.md)). It is produced with the **plain
`baseline` backbone — generic incompressible Navier–Stokes + kinematic wall, no
FSI forcing, no traction** — so it stands on its own regardless of how the A/B
question resolves.

> Driver: `pinnecho.train.observability` (`scripts/observability_sweep.py`,
> console script `pinnecho-sweep`). Raw numbers:
> [`docs/results/observability/sweep.json`](results/observability/sweep.json).
> Figures: [`sweep_vel_relL2_u.png`](results/observability/sweep_vel_relL2_u.png),
> `sweep_pressure_corr.png`, `sweep_vel_relL2_v.png`, `sweep_wss_relL2.png`.

## Question

Doppler ultrasound measures only the **beam-direction** component of velocity. A
PINN must recover the *full* field from this single-component, sparse, noisy
signal plus the incompressible-NS prior. Which acquisition knob actually limits
the reconstruction — the **number of acoustic windows** (angular coverage), the
**SNR**, or the **spatial sampling density**? We sweep each around a base setting
(1 window, ~26 dB, 300 pts/frame) and report held-out error vs. the ground truth.

## Setup

- Backbone `baseline` (no FSI), 2000 Adam + 200 L-BFGS steps, float32, CPU.
- Windows: apical (default) + parasternal-like angled views (matching the window
  convention used throughout the repo). Every window is still single-component.
- **5 seeds**; mean ± std reported. Held-out metrics: velocity relative L2 for the
  cross-beam `u`, primary `v`, and `speed`; vorticity/WSS/pressure correlation.

## Result

| condition | win | SNR (dB) | pts | `u` relL2 ↓ | `v` relL2 ↓ | speed relL2 ↓ | vort corr ↑ | WSS corr ↑ | p corr ↑ |
|---|---|---|---|---|---|---|---|---|---|
| **windows=1** | 1 | 26 | 300 | **0.982 ±0.004** | 0.718 | 0.714 | 0.582 | 0.428 | −0.154 |
| **windows=2** | 2 | 26 | 300 | **0.855 ±0.036** | 0.760 | 0.690 | 0.565 | 0.271 | −0.032 |
| **windows=3** | 3 | 26 | 300 | **0.749 ±0.040** | 0.811 | 0.712 | 0.598 | 0.205 | 0.241 |
| noise=0.02 | 1 | 34 | 300 | 0.983 ±0.004 | 0.720 | 0.716 | 0.579 | 0.423 | −0.118 |
| noise=0.10 | 1 | 20 | 300 | 0.982 ±0.003 | 0.720 | 0.717 | 0.576 | 0.426 | −0.115 |
| points=150 | 1 | 26 | 150 | 0.986 ±0.009 | 0.701 | 0.707 | 0.579 | 0.417 | −0.028 |
| points=600 | 1 | 26 | 600 | 0.978 ±0.010 | 0.695 | 0.703 | 0.575 | 0.405 | 0.007 |

### 1. Angular coverage is *the* lever for the unobserved velocity component

The primary apical beam is nearly aligned with `y`, so a single window observes
`v` reasonably (relL2 0.72) but leaves the cross-beam `u` almost **unconstrained**
(relL2 **0.982** — essentially no better than predicting zero). Adding windows
drives `u` down **monotonically and well outside seed scatter**:

```
u relL2:  1 window 0.982 ±0.004  →  2 windows 0.855 ±0.036  →  3 windows 0.749 ±0.040
```

By contrast, sweeping **SNR from 20→34 dB** or **density from 150→600 pts/frame**
moves `u` by **<1%** (0.978–0.986) — inside the seed noise. *How many directions
are measured* dominates over how cleanly or how densely each is sampled.

### 2. It propagates to pressure (links to Ablation 3)

Pressure correlation rises with window count in lockstep with the `u` recovery
(−0.15 → −0.03 → **+0.24**), while SNR/density leave it near zero. This is the
same coupling isolated in [`ABLATIONS.md`](ABLATIONS.md) Ablation 3: pressure is
recovered from the (pressure-Poisson) velocity field, so it inherits the
cross-beam observability limit rather than needing an FSI term.

### 3. Not a free lunch — a caveat for gradient quantities

Adding windows is **not** uniformly beneficial:

- the **primary** component `v` relL2 *rises* slightly (0.72 → 0.81) and `speed`
  is roughly flat — extra angled windows redistribute the fit away from the
  well-observed direction;
- **WSS correlation *falls*** (0.43 → 0.27 → 0.21). More interior angular coverage
  does not help — and can hurt — the near-wall gradient. This matches the
  independent observation in Ablation 6 that a second window lowers WSS corr.

So multi-window acquisition specifically buys back the **unobserved interior
velocity direction and global pressure**, not every derived quantity.

## Takeaways

1. **The fundamental bottleneck of single-component Doppler reconstruction is
   angular coverage, not measurement quality or density.** One extra window beats
   any realistic improvement in SNR or sampling. This is consistent with the
   multi-view / triplane iVFM literature and is established here on purely
   physics-regularised (FSI-free) reconstructions.
2. **This result is orthogonal to the FSI question** and survives every circularity
   check, because no FSI forcing or traction is used. It is reportable on its own.
3. **Multi-window gains are quantity-specific**: strong for cross-beam `u` and
   pressure, neutral-to-negative for the primary component and wall shear stress —
   a caveat any acquisition-design recommendation should carry.

## Reproduce

```bash
python scripts/observability_sweep.py --backbone baseline --no-traction \
    --steps 2000 --lbfgs-iters 200 --seeds 0 1 2 3 4 --dtype float32 \
    --out docs/results/observability
```

---

# Plane-coverage observability in 3D (standard echo views)

The 2D sweep above abstracts each window as a point that sees the **whole**
field. A real transthoracic exam does not: it acquires a few standard 2D
**imaging planes**, and each view resolves only the beam component of the fluid
lying in its thin slab (`pinnecho.data.acquisition`). This section asks the 3D
version of the same question — *which standard-view protocol makes which
velocity component observable?* — again on the FSI-free `baseline` backbone.

> Driver: `pinnecho.train.plane_coverage` (`scripts/plane_coverage_sweep.py`,
> console script `pinnecho-coverage`). Raw numbers:
> [`docs/results/coverage/coverage.json`](results/coverage/coverage.json).
> Figures: [`coverage_vel_relL2_speed.png`](results/coverage/coverage_vel_relL2_speed.png),
> `coverage_vel_relL2_v.png`, `coverage_pressure_relL2.png`.

## Setup

- 3D volume-preserving-ellipsoid ground truth (`synthetic_ibfe_frames_3d`), long
  axis `z`. Backbone `baseline`, 1500 Adam + 150 L-BFGS, float32, CPU, **3 seeds**.
- Protocols: `a4c` (single apical plane), `a4c+a2c` (apical biplane), `a4c+plax`
  (apical + parasternal, a *different probe apex*), the full `4-view`
  (`a4c+a2c+plax+psax`), and an **idealized** reference — three whole-volume point
  windows (every fluid point seen from 3 beam angles), the setup the 2D sweep used.
- Held-out per-component velocity relative L2 (`u` lateral-x, `v` lateral-y, `w`
  axial-z) + pressure. Mean over seeds; seed std ≈ 0.001–0.003 unless noted.

## Result

| protocol | views | n_data | `u` relL2 ↓ | `v` relL2 ↓ | `w` relL2 ↓ | speed relL2 ↓ |
|---|---|---|---|---|---|---|
| `a4c` (single plane)        | 1 | 1140 | 0.996 | 0.995 | **0.954** | 0.972 |
| `a4c+a2c` (apical biplane)  | 2 | 2723 | 0.993 | 0.993 | 0.954 | 0.971 |
| `a4c+plax` (apical+parasternal) | 2 | 2280 | **0.975** | 0.995 | 0.960 | 0.973 |
| `4-view`                    | 4 | 4725 | 0.981 | 0.993 | 0.960 | 0.972 |
| point-windows ×3 (idealized)| 3 | 9000 | 0.983 | **0.978** | 0.963 | 0.970 |

### 1. Doppler recovers the beam-aligned component first

Every apical beam is dominated by its `z` (axial) projection, so even a *single*
apical plane recovers `w` best (relL2 **0.954**) while leaving the two lateral
components essentially unconstrained (`u`, `v` ≈ 0.99 — no better than zero). The
apical biplane (`a4c+a2c`) shares that same apex/axial orientation, so it does
**not** improve any component beyond seed scatter: more apical data ≠ more
directions.

### 2. A complementary beam angle is what unlocks a lateral component

Adding a *parasternal* window (`plax`, probe moved to +x so the beam carries an
`x`-component) is what makes the lateral-`x` component observable: `u` drops
**0.996 → 0.975**, a ~0.02 shift ≫ the ~0.003 seed std. The `4-view` set keeps
`u` recovered (0.981). This is the 3D restatement of the 2D finding — *how many
distinct beam directions* you measure, not how much apical data you collect.

### 3. The `y`-lateral component is structurally unobservable from standard views

The crux: `v` (lateral-`y`) stays ≈ 0.993–0.995 for **every** planar protocol,
including the full 4-view set — none of the standard transthoracic windows
insonify with a strong `y`-beam-component. Only the idealized whole-volume
windows (which include a lateral `+y` apex) begin to recover it (`v` **0.978**).
So the standard A4C/A2C/PLAX/PSAX set leaves one entire velocity direction
under-determined; closing that gap needs a beam orientation the standard views do
not provide — not better SNR or more frames.

### Caveat — absolute error is compute-limited, the *ordering* is the result

At a CPU-affordable budget (1500+150 steps) the 3D single-component inverse does
not converge to a good absolute reconstruction: `speed` relL2 sits at ≈ 0.97 for
**all** protocols (within ~0.002 seed scatter), because it is dominated by the
components no standard view observes. The robust, physically-interpretable signal
is the **per-component ordering** (axial recovered first; lateral-`x` unlocked by
a complementary apex; lateral-`y` only by non-standard coverage), which is well
above seed noise. Definitive magnitudes need a longer/GPU schedule — the same
compute gate that applies to the real-FSI step (see [`ABLATIONS.md`](ABLATIONS.md)).

## Takeaways

1. **3D intraventricular reconstruction from standard Doppler views is
   coverage-limited, component by component**: axial (beam-aligned) is recovered
   first, a complementary probe apex is required for each additional lateral
   direction, and the `y`-lateral component is not observable from the standard
   A4C/A2C/PLAX/PSAX set at all.
2. This extends the 2D angular-coverage result into explicit 3D anatomy and is,
   like it, **FSI-free** and immune to the circularity caveats.
3. It gives a concrete acquisition-design reading: adding *angularly complementary*
   windows (not more of the same apical view, and not more SNR/frames) is what buys
   observability — and some directions require windows outside the standard set.

## Reproduce

```bash
python scripts/plane_coverage_sweep.py --backbone baseline --no-traction \
    --steps 1500 --lbfgs-iters 150 --seeds 0 1 2 --n-fluid 3000 --dtype float32 \
    --out docs/results/coverage
```
