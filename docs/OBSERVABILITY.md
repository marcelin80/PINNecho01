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
