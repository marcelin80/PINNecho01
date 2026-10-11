# Next measured volume: OSU Zenodo 12515230 (not opened)

We have **not** downloaded or reconstructed these files. This note is from
the paper, the Zenodo record, and the GitHub README. One rest subject +
one stress subject. Case demonstration, not a cohort result.

| | |
|---|---|
| Record | [10.5281/zenodo.12515230](https://doi.org/10.5281/zenodo.12515230) (v2, CC BY 4.0) |
| Paper | Arshad et al., *Magn Reson Med* 2024; 92:1248–1262. [10.1002/mrm.30123](https://doi.org/10.1002/mrm.30123) |
| Code | [OSU-MR/motion-robust-CMR](https://github.com/OSU-MR/motion-robust-CMR) (MATLAB ≥ 2019, CORe) |
| Why first | Measured 3-component velocity after their recon. Not a CFD GT. License is clear. Fail cheap. |

Zenodo files (do **not** grab the 5.1 GB cine first):

| file | size | use |
|---|---|---|
| `4D_Flow_Cartesian_Dataset_11.mat` | 807 MB | **rest 4D-flow, first download** |
| `Exercise_Stress_4D Flow_Cartesian_Dataset_5.mat` | 696 MB | stress, after rest passes |
| `3D Cine Cartesian Dataset 1.mat` | 5.1 GB | wall motion / LV mask only if rest velocity is usable |

The `.mat` files are self-gated expiratory k-space, 20 cardiac bins — **not**
a velocity NPZ. Study IV/V live under
`3D cine-4D flow MRI Reconstruction (Study III IV V)/` in the GitHub repo.
Paper: recon ~11 min; FOV is **whole heart + aortic arch**
(~1.3–2.3 mm). Quantitative check in the paper is **ascending aorta vs 2D-PC**,
not an LV jet. LV must still be shown to sit in the volume (same trap as
aortic-only 4D-flow).

Charité (Vellguth / Wiegand phantom + volunteer 4D-flow) is **inquiry only**.
Do not build their CFD. Draft: [`CHARITE_INQUIRY.md`](CHARITE_INQUIRY.md).
If OSU mass-conservation / noise floor fails, ask them for the measured
25-frame LV 4D-flow (1.96×1.96×2.8 mm) before writing a Fluent deck.

## Five-step gate (half-day, rest only)

Do this on the laptop that has MATLAB. Do not stop the Corrado A train.

1. **Download rest k-space only** (`4D_Flow_Cartesian_Dataset_11.mat`, 807 MB)
   plus clone `OSU-MR/motion-robust-CMR`. Cite CC BY 4.0 / Arshad 2024.
2. **Reconstruct** with the official Study IV script (CORe). Expect a
   20-frame 3-component velocity (+ magnitude). Paper wall-clock ~11 min.
   If MATLAB is missing, stop — do not rewrite CORe in Python.
3. **LV cavity, not aorta.** Segment or threshold a cavity mask that includes
   the LV chamber. If the usable mask is aortic arch / LVOT only, **fail**
   (wrong anatomy for this project, same as 4DFlowNet).
4. **Pack SI public volume.** `velocity (T,Nx,Ny,Nz,3)` m/s, `mask`, `spacing`,
   `origin`, `times`. `f = 0`, `p = 0`. Run
   `scripts/public_volume_stasis.py` **without** `--train`. Record
   `cavity_ml`, `speed_mean`, `speed_p99`, `jet_ml` at 5/10/15 cm/s.
5. **Fail cheap.** Reject if (a) `div u` / stroke volume is nonsense against
   the paper’s Aao 2D-PC, (b) `speed_p99` sits under 10 cm/s so the clinical
   cut has no jet, or (c) the 10 cm/s remainder lives in the aorta, not the
   LV. Pass → one-seed observability later. Fail → Charité 4D-flow request,
   not CFD.

Wall motion from the 5.1 GB cine is **after** step 5 passes. This is still
not an IBFE oracle and not a thrombus study. One subject.

```bash
# after a velocity+mask NPZ exists — no train
PYTHONPATH=src python3 scripts/public_volume_stasis.py \
    --volume data/osu_4dflow/rest_lv.npz \
    --absolute --speed-thresh 0.10 \
    --out docs/results/public_volume/osu_stasis_gt.json
```
