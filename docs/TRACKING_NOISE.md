# Wall-tracking noise: from a placeholder to an STE-literature preset

**Standalone calibration note — independent of the FSI / circularity discussion.**

Model A's wall boundary condition is a *contour-tracking* estimate of endocardial
velocity (segmented contour, finite-differenced in time). Model B is allowed the
accurate FSI structural interface velocity. The gap between those two walls was
previously a hardcoded `(-8% bias, 10% RMS noise)` pair. That pair was always
labelled temporary (`docs/ABLATIONS.md` conclusion item 6 / `PROJECT_STATUS` §9.4).
This note replaces it with **named, citable presets**.

> Module: `pinnecho.data.tracking_noise`
> (`TrackingNoise`, `PRESETS`, `apply_tracking_noise`).
> Realised as `v_kin = v_fsi * (1 + bias) + N(0, (noise * rms(v_fsi))²)`.

## Presets

| name | bias | noise (RMS) | when to use |
|---|---|---|---|
| `exact` | 0 | 0 | `A_exact` / coincident-wall isolation (Ablation 1, 7, 8) |
| `placeholder` | −0.08 | 0.10 | **default** — reproduces published Ablation 1–6 numbers |
| `ste` | −0.05 | 0.09 | going-forward literature calibration (recommended) |

`placeholder` stays the default so re-running Ablation 1–6 does not silently
change the wall BC. New work should pass `tracking="ste"`.

## Literature the `ste` preset is taken from

**Random scatter (`noise = 0.09`).** Houard et al., *Clin Res Cardiol* (2021)
[PMID 32793957](https://pubmed.ncbi.nlm.nih.gov/32793957/): STE-LV-GLS
test–retest coefficient of variation **8.9%** (ICC 0.94) on a mixed healthy /
heart-failure cohort scanned on separate days. This is the right *scale* for a
contour-tracking wall-velocity perturbation: GLS is a displacement-derived
global wall-motion metric, and Model A's wall velocity is a time derivative of
the same tracked contour. The 8.9% sits at the upper end of the EACVI/ASE
multi-vendor study (Farsalinos et al., *JASE* 2015): inter-observer relative
mean error for three-view GLS **5.4–8.6%**, intra-observer 4.9–7.3%. We take
the test–retest number (not the same-loop intra-observer number) because
Model A in a real study inherits acquisition + analysis + biological scatter.

**Systematic under-estimation (`bias = −0.05`).** Amundsen et al., *JACC* 2006
(STE vs sonomicrometry) reported STE underestimating short-axis shortening at
higher values; long-axis strain was unbiased. Contour finite-differencing plus
temporal smoothing pulls peak wall *speed* the same way. Vendor-dependent
STE-vs-tagged-MRI bias is **not** a single number (Obert et al., *Circ
Cardiovasc Imaging* 2018 saw large direction-dependent relative bias), so we
take a conservative 5% under-estimation rather than claiming a unique truth
offset. The old −8% placeholder was already in this ballpark.

**What this is *not*.** It is not a claim that every vendor / every wall segment
has 5%/9% error. Regional STE CoV is substantially worse (far-field /
posterolateral segments 15–40% in some series). The preset is a *global*
endocardial-velocity error model for the synthetic / IBFE wall point cloud —
the same role the 8%/10% placeholder played, now sourced.

## What did *not* change

- Published Ablation 1 (`A_noisy` vs `A_exact`) still describes the placeholder
  pair. `A_exact` remains the fair no-noise baseline; that conclusion does not
  depend on the 8-vs-9% scatter.
- Ablation 7/8 (band-localized oracle / deliverable 3-way) keep **coincident**
  walls (`tracking=None` / `exact`) to isolate the forcing term. Pass
  `tracking="ste"` on `train_ibfe(..., backbone="baseline")` when you want the
  realistic Model-A wall; FSI backbones (`fsi_informed`, `fsi_param`) always
  keep the accurate FSI wall.
- Doppler *measurement* noise (`DopplerConfig.noise_level`, default 5% of mean
  `|v_beam|`) is a separate knob. The observability sweep already showed that
  SNR is not the lever (20–34 dB moves `u` by <1%).

## API

```python
from pinnecho.data import apply_tracking_noise, resolve_tracking
from pinnecho.train.train import build_toy_dataset
from pinnecho.data import train_ibfe

spec = resolve_tracking("ste")                  # bias=-0.05, noise=0.09
v_kin = apply_tracking_noise(v_fsi, "ste", seed=0)

# Toy 2D (Ablation 1 path) -- default is still placeholder
dataset, lv = build_toy_dataset(config, tracking="ste")

# IBFE path: corrupt Model A only; B/oracle keep the exact FSI wall
train_ibfe(frames, backbone="baseline", tracking="ste")
train_ibfe(frames, backbone="fsi_param")        # exact wall, deliverable B
```

YAML:

```yaml
wall_tracking:
  preset: ste          # or exact / placeholder
  # bias: -0.05        # optional override
  # noise: 0.09
```

## Takeaways

1. The Stage-1 8%/10% pair was already close to the STE test–retest literature;
   `ste` (5%/9%) is a small, sourced correction, not a regime change.
2. `A_exact` (zero tracking error) stays the fair baseline for any claim that
   "FSI wall velocity helps" — that comparison does not care whether the noisy
   arm is 8/10 or 5/9.
3. Use `tracking="ste"` for new A-vs-B runs; leave the default alone when
   reproducing published ablations.
