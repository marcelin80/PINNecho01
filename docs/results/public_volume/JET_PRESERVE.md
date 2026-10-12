# Corrado — jet-preserve lever (observable `|v_beam|` only)

The first `--fast-frac` draft stratified **GT 3-component speed**. That
tells the trainer where the 10 cm/s jet is and is the same class of
circularity as feeding oracle `f`. **Do not run that version.**

The deliverable lever oversamples the Doppler *data* batch on a fast set
defined from **observed `|v_beam|`**. Fluid collocation stays uniform.
Fluid point clouds stay uniform
(`frames_from_public_volume(..., fast_frac=None)`).

## Jet definition (confirm before reading millilitres)

`persist` = cavity voxels with `|u| < cut` for **≥ 50 %** of the cycle.
`jet` = `cavity & ~persist` = above the cut for **more than half** the
cycle. Persist 164.6 + jet 30.4 = cavity 195 mL is just that partition
at 10 cm/s. It is **not** “voxels that exceeded the cut at least once.”
A one-frame spike does not enter the jet.

## Fast set (B) — per-frame quantile, not 5/10/15

`--fast-frac 0.5` (command unchanged) now means:

- **set** = per-frame `|v_beam|` ≥ that frame’s **p80** (`--fast-quantile`,
  default 0.80). Rank, not centimetres. Independent of the eval cuts.
- **batch** = 50 % of the Doppler rows drawn from that set.

Do **not** pass `--fast-thresh 0.10` for B. An absolute 10 cm/s mask on
`|v_beam|` is tied to the 10 cm/s score and the comparison is circular.

Limit, written down before the run: `|v_beam|` cannot see a jet moving
**perpendicular to the beam**. Oversampling the observed scalar will not
invent that component.

`n_data` default is **3072 and does not grow with `--n-windows`**. A
(3 windows) and C (1 window) therefore use the same total Doppler rows;
3-window training dilutes the per-window budget (~1024 vs 3072). That
confound stays unless `--scale-n-data-with-windows` (D).

## What to report (not jet mL alone)

Oversampling can grow jet millilitres with noise (fake fast voxels) or
by starving the slow region. Every cut prints and stores:

- `signed_jet_err` = (pred − GT) / GT
- `jet_dice`, `jet_precision`, `jet_recall`
- `slow_speed_relL2` on the **GT persist** voxels
- `peak_ratio_mean` / `peak_ratio_median` (pred/GT `max_t |u|` on the
  GT jet)

Per-seed `*_peaks.npz` writes `peak_true`, `peak_pred`, `jet_true_010`
so the amplitude-bias check does not need a retrain. A did not save
these; do not rerun A.

## Blur-GT upper bound (no train)

15 cm/s jet = 10.8 mL ≈ 1350 voxels at 2 mm. A slightly smooth field
drops Dice hard. Blur the **GT** to a PINN-like voxel σ and score the
same cuts. If blur Dice ~ 0.6, the 15 cm/s failure is a resolution
ceiling. If blur Dice ~ 0.9 and PINN is ~ 0.56, the PINN is worse than
blur.

```bash
PYTHONPATH=src python3 scripts/public_volume_stasis.py \
    --volume data/corrado_lv_phantom/day1_vipr_30min.npz \
    --blur-sigma 1.0 --cuts 0.05 0.10 0.15 \
    --out docs/results/public_volume/stasis_blur_gt.json
```

## Peak-speed hypothesis (no train once peaks exist)

Leak grows with the cut (A, signed): 5 cm/s −10/−7/−8 %; 10 cm/s
−14/−20/−16 %; 15 cm/s −26/−51/−48 %. Hypothesis: the reconstruction
underestimates peak speed by a roughly constant factor, so the tail
clips first. On the GT jet, if pred/GT peak ratio sits near 0.8 on
every seed, the “leak” is amplitude bias. Check from `*_peaks.npz`.

## Pre-registered B decision (set before looking at B)

A 15 cm/s signed errors: −26 / −51 / −48 %. A whole-field speed relL2:
0.417 / 0.461 / 0.414. A did not record `slow_speed_relL2`.

**Adopt B** only if **all three** hold:

1. 15 cm/s `signed_jet_err` ≥ −0.25 on seeds 0, 1, **and** 2.
2. Seed-wise B `vel_relL2_speed` ≤ that seed’s A value + 0.03.
3. 10 cm/s `jet_precision` does not collapse (no fake-jet millilitres)
   and 10 cm/s `slow_speed_relL2` is not larger than that seed’s A
   `vel_relL2_speed` + 0.03.

**Otherwise** drop the sampling lever. Report: the reconstruction
systematically underestimates the high-speed core. Do not invent a
success criterion after seeing B.

## B outcome: reject

Recorded in [`STASIS_RECON_FASTFRAC.md`](STASIS_RECON_FASTFRAC.md).
15 cm/s signed error −23.3 / **−31.9** / **−26.1 %** — seeds 1 and 2
miss −25 %. Slow region and 10 cm/s precision did not fail. Peak-ratio
median on the GT jet is **~0.80–0.88**. Sampling is folded. The finding
is systematic high-speed-core underestimation.

Do not rerun B. Do not adopt on the 10 cm/s millilitre recovery.

“Windows protect the jet” is still a hypothesis until C exists.

## Commands

Do not overwrite `stasis_recon.json`. Do not rerun A.

```bash
# blur-GT (no train) — run before or beside B
PYTHONPATH=src python3 scripts/public_volume_stasis.py \
    --volume data/corrado_lv_phantom/day1_vipr_30min.npz \
    --blur-sigma 1.0 --cuts 0.05 0.10 0.15 \
    --out docs/results/public_volume/stasis_blur_gt.json

# A. baseline, 3 seeds, 3 cuts  — already recorded; do not rerun
PYTHONPATH=src python3 scripts/public_volume_stasis.py --train \
    --volume data/corrado_lv_phantom/day1_vipr_30min.npz \
    --steps 2000 --n-windows 3 --seeds 0 1 2 --cuts 0.05 0.10 0.15 \
    --out docs/results/public_volume/stasis_recon_w3_seeds.json

# B. |v_beam| per-frame p80 oversample, same seeds / windows / cuts
PYTHONPATH=src python3 scripts/public_volume_stasis.py --train \
    --volume data/corrado_lv_phantom/day1_vipr_30min.npz \
    --steps 2000 --n-windows 3 --seeds 0 1 2 --cuts 0.05 0.10 0.15 \
    --fast-frac 0.5 \
    --out docs/results/public_volume/stasis_recon_fastfrac_seeds.json

# C. 1-window, same n_data=3072, same seeds / cuts (window hypothesis)
PYTHONPATH=src python3 scripts/public_volume_stasis.py --train \
    --volume data/corrado_lv_phantom/day1_vipr_30min.npz \
    --steps 2000 --n-windows 1 --seeds 0 1 2 --cuts 0.05 0.10 0.15 \
    --out docs/results/public_volume/stasis_recon_w1_seeds.json

# D. optional: n_data × windows (dilution control), 3 windows
PYTHONPATH=src python3 scripts/public_volume_stasis.py --train \
    --volume data/corrado_lv_phantom/day1_vipr_30min.npz \
    --steps 2000 --n-windows 3 --seeds 0 1 2 --scale-n-data-with-windows \
    --cuts 0.05 0.10 0.15 \
    --out docs/results/public_volume/stasis_recon_w3_ndata.json
```

After A–C (D if claiming the window lever in a paper), **freeze Corrado**.
Not a thrombus.
