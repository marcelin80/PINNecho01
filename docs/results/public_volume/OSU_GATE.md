# OSU rest #11 — fail-cheap gate (pre-registered)

Corrado A/B/C/blur are **closed**. Do not rerun them. This file locks the
reject rules **before** a painted LV is scored. Do not move them after
seeing millilitres.

Windows CORe (2026-10-12): `rest4d`, `recon=["core"]`, `use_gpu=0`,
4 encodings, **16.10 min**, acceleration 31.47. Background-phase
`maps.jpg` is **not** an LV jet. `outputs.theta*` already have
`cmapx/y/z` applied — the packer maps `θ/π · VENC` and does **not**
re-estimate background phase. Residual static-tissue speed is the
leftover. Copy `rest4d_*_core.mat` to monk and pack without `--mask`.

Paper check is **ascending aorta vs 2D-PC**, not an LV jet
(Arshad et al., *Magn Reson Med* 2024; dataset #11). Aao peak ~180 cm/s
and ref stroke volume **93.6 mL/beat**.

## Retired: LV `speed_p99` ≥ 10 cm/s

That cut **cannot fail**. `peak_speed = max_t |u|` is positively biased
over 20 frames. The placeholder pack already has p99 **35 cm/s**, and
unmasked air hits VENC·√3 ≈ **344 cm/s**. A cursor of 576.1 on
`peak_speed.nii.gz` is a **0–1000 display** unit, not cm/s (and not
mm/s). Physical values stay in the sibling `.npy`.

## Five checks (2026-10-12 revision)

| id | pass | fail | pending |
|---|---|---|---|
| `lv_vs_tissue` | spatial median of per-voxel **time-RMS** \|u\| on an LV mask is ≥ **3×** the same statistic on a static-tissue ROI (chest-wall muscle, not lung) | ratio < 3 | no LV, or no tissue ROI. `legacy_speed_p99` is reported only |
| `time_consistency` | harmonics **1–4** hold ≥ **0.55** of AC power on the spatial-mean `v(t)` | flat spectrum or DC-only (< 0.55) | no LV mask |
| `aao_sv` | \|SV − 93.6\| / 93.6 ≤ **0.40** | relative error > 0.40 | no Aao analysis plane |
| `valve_balance` | \|SV_mitral − SV_aorta\| / max(\|SV\|) ≤ **0.40** | relative error > 0.40 | missing mitral or Aao plane |
| `jet_chamber` | 10 cm/s remainder (`cavity & ~persist`) has **≥ 50 %** of its millilitres inside the LV, and **< 50 %** inside an optional aorta/LVOT *region* | remainder empty, mostly aorta/LVOT, or < 50 % in LV | no LV mask, or the packed volume is LV-only (cannot see aorta contamination). A thin Aao plane is **not** a location mask — it is only for SV |

Why these numbers, locked before a painted LV is scored:

- **3× time-RMS**, not p99 of `max_t`. A one-frame noise spike no longer
  writes the statistic. Static chest-wall residual after cmap is the
  noise floor; an LV that is actually wall, RV, or lung stays near 1×.
- **0.55 in harmonics 1–4.** rfft of T=20 has 10 AC bins; a flat
  spectrum expects ~0.40 in bins 1–4. Cardiac RR modulation piles into
  1–4. DC-only (static residual) is 0.
- **Aao ±40 % and mitral/aortic ±40 %.** Integration cancels zero-mean
  noise. Internal valve balance is the second noise-robust flux check.

Jet is still `cavity & ~persist` at 10 cm/s, persist fraction 0.50 — not
“ever exceeded the cut.”

**PASS** = all five pass. PASS does **not** authorize a thrombus solver
or the 5.1 GB cine. One-seed observability is after PASS.

**FAIL** or no confident LV contour in **30–45 min** → Charité 4D-flow
inquiry ([`CHARITE_INQUIRY.md`](../../data/CHARITE_INQUIRY.md)), **not**
CFD. Do not keep painting.

## Mask protocol (after this gate revision)

1. Open `mag_mean.nii.gz` only. **Hide** `peak_speed.nii.gz` while
   painting. PC-MRA / speed-weighted images are for *finding* the
   chamber, not for tracing a wall (that selects high-speed voxels).
2. Confirm millilitres in ITK-SNAP **Segmentation → Volumes and
   Statistics**. Adult LVEDV is typically ≥ 100 mL.
3. Confirm descending aorta is posterior-left and runs S–I before
   trusting R/L/A/P/S/I. The NIfTI affine is **spacing-only**; labels
   are not scanner anatomy.
4. Also draw a chest-wall **tissue** ROI, a thin **Aao** plane, and a
   thin **mitral** plane.
5. Save `rest_lv_mask_v1.nii.gz` (then `v2` if needed). **Do not
   overwrite.** Do not redraw after reading the gate JSON.
6. The discarded auto-box `43:60 59:69 15:37` (~17×10×22 voxels, ~53 mL,
   stair-step rectangle under the anterior chest-wall stripe) is **not
   LV**. Do not score it. Dark-fill on axial slices can paint lung.

## Laptop commands

After MATLAB CORe (see [`OSU_ZENODO.md`](../../data/OSU_ZENODO.md)):

```bash
# 1. pack whole-heart (magnitude is not LV)
PYTHONPATH=src python3 scripts/convert_osu_core.py \
    --recon path/to/rest11_*_core.mat \
    --spacing-mm 2.3 2.3 2.7 \
    --venc-cm-s 200 \
    --out data/osu_4dflow/rest_wholeheart.npz \
    --qa-dir data/osu_4dflow/qa \
    --gate-json docs/results/public_volume/osu_gate_placeholder.json

# 2. after THIS gate revision: hide peak_speed, draw on mag_mean.
#    versioned outputs only (v1 / v2). 30-45 min then stop.
#    PYTHONPATH=src python3 scripts/osu_box_mask.py \
#        --image data/osu_4dflow/qa/mag_mean.npy \
#        --volume data/osu_4dflow/rest_wholeheart.npz \
#        --box X0:X1 Y0:Y1 Z0:Z1 --percentile 55 \
#        --out data/osu_4dflow/rest_lv_mask_v1.nii.gz
#    A one-slice box is a valve plane. A chest-wall box is the tissue ROI.

# 3. score the five checks
PYTHONPATH=src python3 scripts/osu_failcheap_gate.py \
    --volume data/osu_4dflow/rest_wholeheart.npz \
    --lv-mask data/osu_4dflow/rest_lv_mask_v1.nii.gz \
    --tissue-mask data/osu_4dflow/rest_tissue_mask_v1.nii.gz \
    --aao-mask data/osu_4dflow/rest_aao_mask_v1.nii.gz \
    --mv-mask data/osu_4dflow/rest_mv_mask_v1.nii.gz \
    --out docs/results/public_volume/osu_gate.json
```

Do not overwrite Corrado JSON. Do not `--train` until this file records
PASS. Do not score the auto-box as LV.

## Placeholder pack (2026-10-12, no LV mask)

`rest4d_core.mat` → `rest_wholeheart.npz`. VENC 2.00 m/s.

| | |
|---|---|
| velocity | `(20, 77, 96, 72, 3)` m/s |
| spacing | 2.3 × 2.3 × 2.7 mm |
| `mask_is_lv` | False |
| verdict | **PENDING** (all five, once re-scored) |
| placeholder `speed_p99` | **0.3529 m/s** (35 cm/s) — **not a cut** |
| whole-heart 10 cm/s remainder | **469.85 mL** — not an LV jet |

Paper Aao peak is ~180 cm/s. Whole-heart p99 is not that number. Do not
fail or adopt from this row.

Unmasked `max=3.44` m/s, `p999=3.15` m/s. That is VENC·√3 ≈ 3.46
(three components near ±2 m/s), i.e. low-magnitude phase noise, not a
344 cm/s jet. The magnitude mask already dropped that tail
(placeholder p99 0.35 m/s). Next, after this revision, is a painted LV
**plus** tissue + mitral + thin Aao, then `osu_failcheap_gate`.
