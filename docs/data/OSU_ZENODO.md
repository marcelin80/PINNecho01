# Next measured volume: OSU Zenodo 12515230

Corrado A/B/C/blur are **closed**. Do not rerun them. This is the next
measured 3-component volume. One rest subject (dataset #11 is a
**patient**). Case demonstration, not a cohort. CC BY 4.0.

| | |
|---|---|
| Record | [10.5281/zenodo.12515230](https://doi.org/10.5281/zenodo.12515230) (v2) |
| Paper | Arshad et al., *Magn Reson Med* 2024; 92:1248–1262 |
| Code | [OSU-MR/motion-robust-CMR](https://github.com/OSU-MR/motion-robust-CMR) (MATLAB ≥ 2019) |
| First file | `4D_Flow_Cartesian_Dataset_11.mat` (807 MB, md5 `7a0466a6f173590990d827350c0b34df`) |
| VENC | **200 cm/s** (paper Fig. 4, dataset #11) |
| FOV | whole heart + aortic arch. Spatial res. Study IV: 2.3–3.1 mm |
| Fail-cheap | [`OSU_GATE.md`](../results/public_volume/OSU_GATE.md) (rules locked) |

The `.mat` is self-gated expiratory **k-space**, 20 cardiac bins — not a
velocity NPZ. Do **not** download the 5.1 GB cine first. Do **not**
rewrite CORe in Python. If MATLAB ≥ 2019 is missing, **stop**.

Paper check is **ascending aorta vs 2D-PC**, not an LV jet. Dataset #11
Aao peak ~180 cm/s, so whole-heart `speed_p99` will clear 10 cm/s. That
does not mean the LV has a usable jet.

## Laptop step 1 — MATLAB check, then download + clone

monk (2026-10-12): `command -v matlab` was empty. That is a **stop**,
not a reason to rewrite CORe. Hunt off-PATH before declaring none.
If still empty, **do not curl** the 807 MB k-space; send
[`CHARITE_INQUIRY.md`](CHARITE_INQUIRY.md).

```bash
# stop here if this exits 2 — do not continue to curl
bash scripts/osu_matlab_check.sh
```

Only if that printed a `MATLAB` path:

```bash
cd ~/PINNecho01
mkdir -p data/osu_4dflow
cd data/osu_4dflow

curl -L --fail -o 4D_Flow_Cartesian_Dataset_11.mat \
  "https://zenodo.org/api/records/12515230/files/4D_Flow_Cartesian_Dataset_11.mat/content"
md5sum 4D_Flow_Cartesian_Dataset_11.mat
# expect 7a0466a6f173590990d827350c0b34df

git clone --depth 1 https://github.com/OSU-MR/motion-robust-CMR.git
```

Cite: Arshad et al. 2024; Zenodo 10.5281/zenodo.12515230, CC BY 4.0.

## Laptop step 2 — CORe only (skip CS)

```matlab
cd('.../motion-robust-CMR/3D cine-4D flow MRI Reconstruction (Study III IV V)')
% in main_recon.m set:  recon = ["core"];   opt.use_gpu = 1 if NVIDIA
main_recon
% prompts: save name (e.g. rest11); flow=1; rest=1;
% then pick 4D_Flow_Cartesian_Dataset_11.mat
```

Paper wall-clock ~11 min (GPU). Output:
`outputs.thetaX/Y/Z` (phase, **not** m/s) and `outputs.xHat`.
Saved as `<saveName>_<date>_core/<saveName>_<date>_core.mat`.

## Laptop step 3 — pack whole-heart + QA (no LV claim)

`--mask` must be an LV cavity. A magnitude percentile is a placeholder
and leaves the gate **PENDING**.

```bash
PYTHONPATH=src python3 scripts/convert_osu_core.py \
    --recon path/to/rest11_*_core.mat \
    --spacing-mm 2.3 2.3 2.7 \
    --venc-cm-s 200 \
    --out data/osu_4dflow/rest_wholeheart.npz \
    --qa-dir data/osu_4dflow/qa \
    --gate-json docs/results/public_volume/osu_gate_placeholder.json
```

ITK-SNAP / 3D Slicer: open `data/osu_4dflow/qa/peak_speed.nii.gz`
(needs `pip install nibabel` for the `.nii.gz`; PNGs and `.npy` are
always written). Draw:

- LV cavity → `data/osu_4dflow/rest_lv_mask.nii.gz`
- **thin** Aao analysis plane (one slice, not the whole arch) →
  `data/osu_4dflow/rest_aao_mask.nii.gz`

## Laptop step 4 — fail-cheap (rules locked)

```bash
PYTHONPATH=src python3 scripts/osu_failcheap_gate.py \
    --volume data/osu_4dflow/rest_wholeheart.npz \
    --lv-mask data/osu_4dflow/rest_lv_mask.nii.gz \
    --aao-mask data/osu_4dflow/rest_aao_mask.nii.gz \
    --out docs/results/public_volume/osu_gate.json
```

Reject if (a) Aao stroke volume relative error vs 93.6 mL/beat **> 0.40**,
(b) LV `speed_p99` < 10 cm/s, or (c) ≥ 50 % of the 10 cm/s remainder
lives in the aorta / LVOT, or < 50 % lives in the LV. Exact table:
[`OSU_GATE.md`](../results/public_volume/OSU_GATE.md).

PASS → one-seed observability later. FAIL → Charité 4D-flow request
([`CHARITE_INQUIRY.md`](CHARITE_INQUIRY.md)), **not** CFD.

Wall motion from the 5.1 GB cine is **after** step 4 passes. Still not
an IBFE oracle and not a thrombus.
