# OSU rest #11 — fail-cheap gate (pre-registered)

Corrado A/B/C/blur are **closed**. Do not rerun them. This file locks the
reject rules **before** the laptop CORe recon exists. Do not move them
after seeing millilitres.

Paper check is **ascending aorta vs 2D-PC**, not an LV jet
(Arshad et al., *Magn Reson Med* 2024; dataset #11). Aao peak ~180 cm/s
and ref stroke volume **93.6 mL/beat**. Whole-heart `speed_p99` will
clear 10 cm/s even if the LV has no usable jet.

## Three checks

| id | pass | fail | pending |
|---|---|---|---|
| `speed_p99` | ≥ 10 cm/s **on an LV mask** | LV p99 < 10 cm/s | measured on a magnitude placeholder |
| `aao_sv` | \|SV − 93.6\| / 93.6 ≤ **0.40** | relative error > 0.40 | no Aao analysis plane |
| `jet_chamber` | 10 cm/s remainder (`cavity & ~persist`) has **≥ 50 %** of its millilitres inside the LV, and **< 50 %** inside an optional aorta/LVOT *region* | remainder empty, mostly aorta/LVOT, or < 50 % in LV | no LV mask, or the packed volume is LV-only (cannot see aorta contamination). A thin Aao plane is **not** a location mask — it is only for SV. |

Jet is still `cavity & ~persist` at 10 cm/s, persist fraction 0.50 — not
“ever exceeded the cut.”

**PASS** = all three pass. PASS does **not** authorize a thrombus solver
or the 5.1 GB cine. One-seed observability is after PASS.

**FAIL** → Charité 4D-flow inquiry ([`CHARITE_INQUIRY.md`](../../data/CHARITE_INQUIRY.md)),
**not** CFD.

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

# 2. draw LV cavity + a *thin* Aao plane on peak_speed.nii.gz (ITK-SNAP)

# 3. score the three checks
PYTHONPATH=src python3 scripts/osu_failcheap_gate.py \
    --volume data/osu_4dflow/rest_wholeheart.npz \
    --lv-mask data/osu_4dflow/rest_lv_mask.nii.gz \
    --aao-mask data/osu_4dflow/rest_aao_mask.nii.gz \
    --out docs/results/public_volume/osu_gate.json
```

Do not overwrite Corrado JSON. Do not `--train` until this file records
PASS.
