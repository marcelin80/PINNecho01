# DUPLEX — scoped, not a volume substitute

Sun, Vixège, Faraz, Mendez, Nicoud, Garcia, Bernard,
*IEEE Trans Ultrason Ferroelectr Freq Control* 2022; 69(3).
[Project](https://humanheart-project.creatis.insa-lyon.fr/duplex.html) ·
Girder collection `DUPLEX` (public, no login).

This is **not** a third LV-cavity 4D-flow volume. It does not replace
OSU or Charité for window-count observability or jet millilitres.
It is a possible probe of one weakness: PINNecho’s measurement model
is an ideal beam projection. Lock the question **before** any PINN run.

## One-line scope

DUPLEX can test **simulated Doppler degradation vs an ideal snapshot**.
It cannot test the remaining core result (window count) or Eulerian
jet / stasis millilitres.

## What the public dump actually is

Inspected 2026-10-12 (Girder listing + `matlab_code.zip` + one `.mat`,
`patient_04_scenario_#1.mat`). Do not move these facts after a train.

| | |
|---|---|
| Layout | `patient_01`…`patient_04` × scenarios `#1`–`#4` = **16** `.mat` (~556 MB) + `code/matlab_code.zip` |
| Paper vs dump | Paper: 5 virtual patients × 4 scenarios = 20. Public collection has **4** patients. `patient_05` is absent |
| Format | MATLAB structs: `bmode_real`, `bmode_simu`, `doppler_real`, `doppler_simu`, `fov` |
| Doppler | **Numeric velocities** (m/s) plus simulated **I/Q** (`doppler_simu.pol.iqs_slowtime`). Not RGB-only. Viewer colour-maps them |
| Frames | Paper: 16. This file: **9** (`param.nb_imgs = 9`) |
| Geometry | Polar sector (`TH`, `R`) + Cartesian resample. A3C, ~12 cm depth. Beam direction varies with angle |
| Probe | 2.5 MHz, 64-element phased array, PRF 7000 Hz, packet 8. Nyquist `c·PRF/(4·fc)` = **1.078 m/s** (~1.1) |
| LV mask | Yes: per-frame `LVs` (polar and Cartesian) |
| Wall / myocardium mask | **Not** in the `.mat`. No wall-velocity field. Model A no-slip is not supported as released |
| CFD 2-component GT | **Not** in the `.mat`. No `(u,w)` / motion-field key. You cannot score a PINN vector field against Chnafa from this download |
| License | Girder `public: true`, empty `meta`. No CC statement. Cite the paper. Do not claim CC BY. HHP FAQ: abide by project terms |
| Flow n | **One** patient-specific 3-D CFD (Chnafa et al.). Four echo *templates*. Registration onto the template **breaks mass conservation** (paper §II-B2) |

Scenarios (paper):

1. Ideal snapshot — stationary flow, no myocardial motion. Estimated Doppler should match **radial CFD**. That radial CFD is not in the file.
2. Unsteady flow (acceleration during the packet).
3. + wall clutter.
4. Focused vs diverging transmit (still snapshot-like).

`doppler_simu.pol.velocities` on the inspected file sit on **±1.078 m/s**
(the Nyquist rails). The paper states filling exceeds 1.1 m/s and shows
alias wrap. You cannot read an unwrapped CFD peak from the packed map.
The clinical template in the same file has `VNyquist = 0.644` m/s and
Cartesian `|v|` up to ~1.63 m/s — that is the *real* duplex, not GT.

## Fits / does not fit

| Question | DUPLEX |
|---|---|
| Do more windows improve the field? | **No.** Single A3C sector |
| Does the recon systematically shave the jet? | Partial, area only, 2-D, ~9–16 frames |
| Do metrics hide clinically important loss? | Partial |
| How much does *simulated* Doppler degradation worsen a recon? | Only if a vector GT exists. **It is not in the public `.mat`** |
| Is the projection measurement model right? | Same: needs radial CFD vs scenario `#1`. Paper already reports R² = 0.92 for *their* estimator. We cannot rerun that comparison from the dump |
| Cohort generalization | **No.** One CFD, healthy-ish templates |
| FSI / `f` | N/A (already closed) |

The 2-D incompressibility mismatch is **worse** than “a 3-D slice has
∇·u ≠ 0 in-plane.” They warp the slice onto the echo mesh and say
mass conservation is not preserved. Scenario `#1` error can already be
dominated by that. Then degradation deltas look small.

## Pre-registered rules (before any PINN number)

1. **Do not download to “get a volume.”** OSU / Charité stay the
   in-vivo volume path. Corrado stays the observability / jet-mL record.
2. If used at all, the only claim allowed is
   **Δ(scenario k − scenario #1)** on a pre-registered score.
   Write `#1` first. If `#1` is already poor, stop: that is the 2-D /
   registration ceiling. Do **not** title the paper as robustness to
   clutter / acceleration / diverging waves.
3. Do not quote window-count, jet mL, persist Dice, or a cohort from
   DUPLEX.
4. Do not treat `doppler_real` as flow GT. It is the clinical template
   duplex (already Nyquist-limited).
5. Do not invent a CFD vector field from the packed Doppler. If the
   experiment needs Chnafa `(u,w)` or radial CFD, **ask the authors**
   (Olivier Bernard / Damien Garcia). Until that file exists, there is
   no PINN train.
6. Degradation is **simulated**. The claim ceiling is robustness to
   SIMUS + their clutter model, not to bedside echo.

## No-train checklist (closed 2026-10-13)

| check | result |
|---|---|
| File format | `.mat` structs; Doppler is m/s + I/Q, not RGB |
| GT range | No packed CFD vector. Simu Doppler sits on ±1.078 m/s |
| CFD model count | **One** Chnafa field; 4 public echo templates (paper said 5) |
| License | Public Girder, no CC name. Cite the paper |
| Projection consistency | **Cannot.** Radial CFD is not in the file |
| Planar divergence | **Cannot.** Need `(u,w)` |
| Peak vs Nyquist | Paper: filling exceeds 1.1 m/s. Packed map is already wrapped. Unwrapped peak unknown |

**Verdict:** the public dump is **not usable** as a PINN robustness GT.
Contact for the missing field is off the essential path. Fallback:
apply **our** synthesizer to Corrado
([`DEGRADE_CORRADO.md`](../results/public_volume/DEGRADE_CORRADO.md)).
Published iVFM-PINN numbers are not confirmed to be the same 20 sequences;
do not put them in the same table.

## If the authors send the missing field

Then, and only then: scenario `#1` is the measurement-model baseline
(PINN input = ideal radial Doppler; score vs planar registered CFD).
Scenarios `#2`–`#4` are the degradation ladder. Same PINN, same mask
(`LVs`). Report deltas, not absolute “we reconstructed LV flow.”

## Do not do

- Train `fsi_informed` (no `f`).
- Use DUPLEX CFD (even if sent) as thrombus / CHF GT.
- Rewrite SIMUS in Python.
- Score after peeking at `#2`–`#4` and then moving the `#1` bar.
