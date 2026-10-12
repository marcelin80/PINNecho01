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

The `.mat` is self-gated expiratory **k-space**, 20 cardiac bins — not a
velocity NPZ. Do **not** download the 5.1 GB cine first. Do **not**
rewrite CORe in Python. If MATLAB ≥ 2019 is missing, **stop**.

Paper check is **ascending aorta vs 2D-PC**, not an LV jet. Dataset #11
Aao peak ~180 cm/s, so `speed_p99` will clear 10 cm/s. That does not
mean the LV has a usable jet.

## Laptop step 1 — MATLAB check, then download + clone

```bash
# stop here if this prints nothing
command -v matlab || echo "NO MATLAB — do not rewrite CORe"

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

## Step 3–4 — pack SI volume (after recon exists)

`--mask` must be an LV cavity. A magnitude percentile is a placeholder
and **fails** the LV gate until a real cavity is drawn.

```bash
PYTHONPATH=src python3 scripts/convert_osu_core.py \
    --recon path/to/rest11_*_core.mat \
    --spacing-mm 2.3 2.3 2.7 \
    --venc-cm-s 200 \
    --out data/osu_4dflow/rest_wholeheart.npz

# after an LV mask exists
PYTHONPATH=src python3 scripts/public_volume_stasis.py \
    --volume data/osu_4dflow/rest_lv.npz \
    --absolute --speed-thresh 0.10 \
    --cuts 0.05 0.10 0.15 \
    --out docs/results/public_volume/osu_stasis_gt.json
```

## Step 5 — fail cheap

Reject if (a) Aao stroke volume is nonsense vs the paper’s 2D-PC
(dataset #11 ref 93.6 mL/beat), (b) `speed_p99` < 10 cm/s, or (c) the
10 cm/s remainder lives in the aorta / LVOT, not the LV chamber.
Pass → one-seed observability later. Fail → Charité 4D-flow request
([`CHARITE_INQUIRY.md`](CHARITE_INQUIRY.md)), **not** CFD.

Wall motion from the 5.1 GB cine is **after** step 5 passes. Still not
an IBFE oracle and not a thrombus.
