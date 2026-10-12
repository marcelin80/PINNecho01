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
rewrite CORe in Python. MATLAB ≥ 2019 can be **Windows**. Do not install
a Linux copy just because PINNecho runs in a monk venv.

Paper check is **ascending aorta vs 2D-PC**, not an LV jet. Dataset #11
Aao peak ~180 cm/s, so whole-heart `speed_p99` will clear 10 cm/s. That
does not mean the LV has a usable jet.

## Step 1 — CORe host is Windows MATLAB (not a Linux reinstall)

monk PATH has no `matlab`. The PINNecho venv does not need one.
Install Linux MATLAB only if the Windows copy is missing, older than
R2019, or you cannot boot/reach that machine. Octave is not a substitute.

On **Windows** (PowerShell). md5 must match before `main_recon`.

```powershell
mkdir $HOME\PINNecho01-osu -Force | Out-Null
cd $HOME\PINNecho01-osu
curl.exe -L --fail -o 4D_Flow_Cartesian_Dataset_11.mat `
  "https://zenodo.org/api/records/12515230/files/4D_Flow_Cartesian_Dataset_11.mat/content"
Get-FileHash 4D_Flow_Cartesian_Dataset_11.mat -Algorithm MD5
# expect 7A0466A6F173590990D827350C0B34DF
# Windows often has no git — zip is enough
curl.exe -L --fail -o motion-robust-CMR.zip `
  "https://github.com/OSU-MR/motion-robust-CMR/archive/refs/heads/main.zip"
Expand-Archive -Force motion-robust-CMR.zip -DestinationPath .
```

Cite: Arshad et al. 2024; Zenodo 10.5281/zenodo.12515230, CC BY 4.0.

## Step 2 — CORe only on Windows (skip CS)

```matlab
% Current Folder MUST be this Study III–V directory (else datestr8601 is missing)
cd('C:\Users\marce\PINNecho01-osu\motion-robust-CMR-main\3D cine-4D flow MRI Reconstruction (Study III IV V)')
assert(exist('functions/datestr8601.m','file')==2, 'cd to the Study III-V folder first')
% edit main_recon.m line ~56:  recon=["cs","core"];  →  recon=["core"];
% default opt.use_gpu=1; set 0 if this Windows box has no NVIDIA
main_recon
% prompts: name rest4d (or rest11);  flow=1;  rest=1;
% file picker: C:\Users\marce\PINNecho01-osu\4D_Flow_Cartesian_Dataset_11.mat
```

Paper wall-clock ~11 min (GPU). Output:
`outputs.thetaX/Y/Z` (phase, **not** m/s) and `outputs.xHat`.
Saved as `<saveName>_<date>_core/<saveName>_<date>_core.mat`.
Copy **that** file to monk (`~/PINNecho01/data/osu_4dflow/`). The 807 MB
k-space can stay on Windows.

## Laptop step 3 — pack whole-heart + QA on monk (no LV claim)

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
