# Paper draft — evaluation, not a patient GT

Venue: methods / evaluation. No contact-required data. No patient
accuracy claim. Numbers below are frozen; do not rerun Corrado A/B/C/blur.

Working title (not “windows help”, not “FSI helps”):

**Evaluation metrics hide jet loss in a Doppler PINN: a measured LV
phantom and a locked degradation fallback**

## Abstract

Physics-informed networks reconstruct intraventricular velocity from
single-component Doppler-like samples. On a manufactured FSI field the
usual losses can look solved while the forcing is a rearrangement of
`∇p`. On a measured in-vitro LV 4D-flow volume (Corrado et al., rigid
silicone + pump, `f = 0`) a three-window PINN reaches speed relL2
~0.42 and persist Dice ~0.96, yet the 10 cm/s Eulerian jet leaks
4–6 mL (24.2–26.2 vs 30.4 mL) and the 15 cm/s core Dice is ~0.56.
Oversampling the fast beam component (B) is rejected on a
pre-registered 15 cm/s bar; peak-speed ratio on the GT jet stays
~0.80–0.88. A 1-voxel blur of the GT caps 15 cm/s Dice at 0.76: the
PINN sits below that ceiling. No public patient volume with a cavity
mask is available without inquiry. A public duplex simulation (DUPLEX)
has no packed CFD vector field, so it is not a reconstruction GT.
Contact-free robustness, if claimed, must use our own synthesizer on
the measured phantom and say so. We do not claim patient accuracy,
clinical generalization, or an FSI benefit.

## 1. Introduction

Colour Doppler sees one component along the beam. PINNs and iVFM
recover a 2-D or 3-D field by adding physics. The tempting story is
that more windows, or an FSI term, close the gap to 4D-flow MRI.

That story is easy to fake. Manufactured solutions set
`f ≈ ∇p` and then “discover” that giving `f` to the network restores
`p`. Persist Dice on a stasis-rich cavity scores the slow sea and
misses the jet. A paper that only reports relL2 and persist Dice can
look done while the 15 cm/s core is gone.

This draft is an evaluation paper on **contact-free** resources:

| resource | role |
|---|---|
| Manufactured LV (Stage 1) | circularity of oracle `f` |
| Corrado PC VIPR 30 min | measured 3-component GT, jet / stasis, windows |
| DUPLEX public `.mat` | inspected; **not** a PINN GT |
| Our Doppler synthesizer | fallback degradation; labelled ours |

Not in the essential path: Charité, Delft, OSU masks, MIMIC
credentialing, author emails.

## 2. Related work (short)

- iVFM / physics-constrained iVFM (Assi, Garcia, …): 2-D polar,
  planar continuity. Same lineage, different data contract. Published
  errors (e.g. 0.3–12 % on a CFD slice) are **not** confirmed to be
  the DUPLEX 16-file dump. Do not share a table.
- Corrado et al., *Magn Reson Med* 2021, OSF 2naw4: rigid LV + PIV.
  We use Day 1 PC VIPR 30 min as MRI GT. PIV FOV misses the jet.
- Sun et al., IEEE TUFFC 2022, DUPLEX: SIMUS duplex from one Chnafa
  CFD registered onto echo templates. Public files: B-mode, Doppler
  velocities, I/Q, `LVs`. No `(u,w)`.
- 4D-flow patient dumps with an LV mask remain inquiry-only.

## 3. Methods

### 3.1 Public-volume contract

Velocity `(T,Nx,Ny,Nz,3)` m/s, cavity mask, spacing, times.
`forcing_fluid = 0`, `pressure = 0` unless measured. Do not train
`fsi_informed` on these volumes.

### 3.2 Corrado volume

20 frames, T = 0.75 s, 2 mm, 71×53×87, cavity **195.0 mL**.
`speed_p99` **0.261 m/s**. Persist = `|u| <` cut for ≥ 50 % of
frames. Jet = `cavity & ~persist` (not “ever exceeded”).

| cut | GT jet |
|---|---|
| 5 cm/s | 79.7 mL |
| 10 cm/s | **30.4 mL** |
| 15 cm/s | 10.8 mL |

### 3.3 Measurement model used for A/B/C

Each window: `v_beam = u · e_beam` plus `noise_level = 0.05`.
Default windows: apical `−z`, `lat-x`, `lat-y`. This is an **ideal
projection**, not SIMUS. That is the weakness DUPLEX would have
tested if it had packed a vector GT.

### 3.4 Network

Baseline incompressible PINN, data / PDE / BC weights 10 / 1 / 10,
`n_data = 3072` (does not scale with windows), 2000 Adam + 50
L-BFGS, 3 seeds. B: per-frame `|v_beam|` p80 oversample, same
`n_data`. C: 1 window, same `n_data`.

### 3.5 Metrics (use these, not persist Dice as success)

- speed / component relL2
- jet millilitres and signed `(pred − GT) / GT` at 10 and 15 cm/s
- jet Dice, precision, recall
- `peak_speed_report`: median pred/GT of `max_t |u|` on the **GT** jet
- blur-GT (σ = 1 voxel, no train) as a smoothness ceiling

Do not title persist Dice ~0.96 or 5 cm/s Dice ~0.84 as success.

### 3.6 DUPLEX inspection (no train)

Public Girder, no login. 4×4 `.mat` (paper said 5×4). Doppler is
numeric + I/Q. `LVs` present. **No CFD `(u,w)`.** Projection
consistency and planar divergence **cannot** be computed. License:
public, no CC name. Verdict: not a reconstruction GT.
Details: [`data/DUPLEX.md`](data/DUPLEX.md).

### 3.7 Degradation fallback (not yet run)

[`results/public_volume/DEGRADE_CORRADO.md`](results/public_volume/DEGRADE_CORRADO.md).
Our synthesizer on Corrado. Clinical Nyquist 1.078 m/s is likely
**inactive** (p99 0.261). Write `ideal` first; report Δ only.
`alias_ours_15cms` must be labelled ours.

## 4. Results (frozen)

### 4.1 Circularity (manufactured, closed)

Oracle `f` tracks `∇p`. `fsi_param` without true `f` matches
baseline. Do not claim an FSI benefit. See [`ABLATIONS.md`](ABLATIONS.md).

### 4.2 Observability on Corrado (supporting, not the title)

3-window, 3-seed, 2000 steps ([`CORRADO_OBSERVABILITY.md`](results/public_volume/CORRADO_OBSERVABILITY.md)):

| windows | speed | u | v | w |
|---|---|---|---|---|
| 1 | 0.506 | 0.728 | 0.898 | **0.260** |
| 2 | 0.432 | **0.409** | 0.834 | 0.339 |
| 3 | **0.418** | 0.423 | **0.690** | 0.356 |

Cross-beam components move. `w` relL2 gets worse as windows are
added. Do not title “windows restore the field.”

### 4.3 Jet loss on the same reconstructions

A, 3 windows × 3 seeds
([`STASIS_RECON_SEEDS.md`](results/public_volume/STASIS_RECON_SEEDS.md)):

| cut | GT | pred mean (range) | Dice mean | signed err |
|---|---|---|---|---|
| 5 cm/s | 79.7 | 73.2 (72.0–74.5) | 0.84 | −10 / −7 / −8 % |
| 10 cm/s | 30.4 | **25.3 (24.2–26.2)** | **0.81** | −14 / −20 / −16 % |
| 15 cm/s | 10.8 | **6.3 (5.3–8.0)** | **0.56** | −26 / −51 / −48 % |

C (1 window): 10 cm/s **23.7 mL**. Windows help the 10 cm/s remainder
a little; they do not save the 15 cm/s core.

B (`|v_beam|` p80): **rejected**. 15 cm/s signed −23.3 / −31.9 /
−26.1 % (bar was ≥ −25 % on all seeds). 10 cm/s 28.6 mL is not a
fake jet; the core still fails.

Peak ratio (B, GT jet, same `peak_speed_report`):

| cut | seed 0 | 1 | 2 |
|---|---|---|---|
| 10 cm/s | 0.879 | 0.850 | 0.821 |
| 15 cm/s | 0.838 | 0.835 | 0.798 |

**~0.83.** Do not rerun to reproduce this table.

### 4.4 Blur-GT ceiling (no train)

σ = 1 voxel on GT velocity:

| cut | Dice | peak ratio |
|---|---|---|
| 10 cm/s | 0.911 | 0.892 |
| 15 cm/s | **0.760** | 0.883 |

PINN A/B are below both Dice caps. 15 cm/s is partly resolution
(blur recall 0.62) and partly misplacement (blur precision 0.98 vs
B 0.74–0.85). Peak 0.88–0.90 vs PINN 0.80–0.88: most of the
amplitude bias is compatible with one-voxel smoothness.

### 4.5 PIV and echo slabs (negative controls)

Registered PIV FOV is 73 mL of stasis sea; MRI jet 0.0 mL in the
overlap. A4C slab covers 73 % of the jet and 62 % of the cavity
(thick 0.12×extent slab). Neither is a jet GT.

### 4.6 DUPLEX / degradation

No PINN number. Public dump cannot score a vector recon.
Fallback ladder is locked and **not run**. If the paragraph is
needed, run `ideal` first.

## 5. Discussion

The result that survives contact-free resources is evaluative:
**relL2 and persist Dice hide a systematic jet under-estimate on a
measured cavity.** Sampling and extra windows do not restore the
15 cm/s core. A blur ceiling shows the PINN is worse than
one-voxel smoothness on location.

Observability is real and unsurprising. It is a subsection.

FSI is closed in the opposite direction: manufactured `f` is
circular; deliverable `fsi_param` does not beat baseline without
true `f`.

## 6. Limitations (write these)

- No patient 3-component GT. No clinical accuracy or generalization.
- Corrado is a rigid pump, not a failing / dilated LV.
- A/B/C measurements are ideal projections, not bedside Doppler.
- DUPLEX cannot fill that gap from the public files.
- A synthesizer fallback is our degradation model. Corrado is too
  slow for a 1.1 m/s Nyquist wrap (`p99` 26 cm/s).
- No thrombus, no Lagrangian residence time on a leaking field.
- iVFM-PINN published errors are not a same-data bake-off.
- Image-IBFE beats exported before Decision 9 treat the image as the
  unstressed reference. Diastolic filling is unphysiological; those
  beats are systole/ejection only. Do not use them to claim an E-wave,
  IVPG, or vortex-ring result. [`DECISION_9.md`](DECISION_9.md).
- The IBFE `pipeline-validation` series is wall-only
  (`eulerian_present = 0`). Oracle A/B on true IBFE `f` was **not**
  done. Ablation 7 (band-localized oracle) already hurt pressure.
  [`IBFE_PIPELINE_LIMITATIONS.md`](IBFE_PIPELINE_LIMITATIONS.md).

## 6b. Paper strength (locked, 2026-10-28)

Observability is not new (triplane iVFM). Persist Dice ~0.96 follows
from persist occupying ~84% of the cavity — do not lead with it as
the metric-failure exhibit. Jet under-estimate overlaps PINN spectral
bias (Rahaman et al., ICML 2019; Wang, Yu and Perdikaris, *J. Comput.
Phys.* 2022; Wang, Wang and Perdikaris, *CMAME* 2021). That is an
overlap with a known smoothing prior, not a newly discovered PINN
law. Fourier features (Tancik et al., 2020) are **off by default**
here because they corrupt derivatives; they were not ablated on the
Corrado jet. Data: one rigid phantom, three seeds, one method.

Defensible contribution: quantify global-metric vs downstream-task
mismatch and propose a protocol. Needs the same direction on a
classical baseline and DUPLEX, ≥ 5 seeds. Journal passage after
review is unlikely (probability unknown). Surest product: preprint
plus public code and evaluation tools.

**Stop before seeing millilitres:** 4–6 week cap. If jet loss does
not reproduce on a classical baseline or DUPLEX, cut to a technical
note. Do not train DUPLEX. Do not rerun Corrado A/B/C.

## 7. What this paper is not

A reconstruction challenge win. An FSI paper. A clinical validation.
A licence to start a thrombus solver.

## 8. Next (optional, not essential)

1. If the draft needs a degradation paragraph: one-seed `ideal` then
   `snr10` / `sparse50` on Corrado. Drop `alias_clinical` if wrap
   fraction < 0.01.
2. Preprint + code, then optionally tell neighbouring groups what
   the metric failure is. That is not a data request.
3. MIMIC-IV-Echo only as an input demo (no GT), and only with spare
   time and a credentialed reference.
4. Charité / Delft / OSU painting stay off the essential path.

## Citations to keep at hand

- Corrado et al., *Magn Reson Med* 2021; OSF 2naw4.
- Sun et al., IEEE TUFFC 2022 (DUPLEX).
- Chnafa, Mendez, Nicoud, *Comput Fluids* 2014 (CFD behind DUPLEX).
- Assi et al. / Garcia iVFM papers (lineage, not same-data scores).
- Arshad et al., *Magn Reson Med* 2024; Zenodo 12515230 (mentioned
  as inquiry-only, not used).
- Rahaman et al., *On the Spectral Bias of Neural Networks*, ICML
  2019, PMLR 97:5301–5310.
- Wang, Yu and Perdikaris, *When and why PINNs fail to train: A
  neural tangent kernel perspective*, *J. Comput. Phys.* 449:110768
  (2022).
- Wang, Wang and Perdikaris, *On the eigenvector bias of Fourier
  feature networks*, *Comput. Methods Appl. Mech. Eng.* 384:113938
  (2021).
- Tancik et al., *Fourier Features Let Networks Learn High Frequency
  Functions in Low Dimensional Domains*, NeurIPS 2020 (available here;
  off by default).
