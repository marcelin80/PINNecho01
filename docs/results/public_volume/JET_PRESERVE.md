# Corrado — jet-preserve lever (observable `|v_beam|` only)

The first `--fast-frac` draft stratified **GT 3-component speed**. That
tells the trainer where the 10 cm/s jet is and is the same class of
circularity as feeding oracle `f`. **Do not run that version.**

The deliverable lever oversamples the Doppler *data* batch on
`|v_beam| >= fast_thresh` (the scalar the PINN is trained on). Fluid
collocation stays uniform. Fluid point clouds stay uniform
(`frames_from_public_volume(..., fast_frac=None)`).

`--fast-frac` on `scripts/public_volume_stasis.py` now means `|v_beam|`.

The 1-window 23.7 mL vs 3-window 26.2 mL contrast is **one seed**. Treat
“windows protect the jet a little” as a **hypothesis** until a 3-seed
paired comparison exists. Score the same `vel_pred` at 5 / 10 / 15 cm/s
(`--cuts`). Jet millilitres are a threshold remainder, not a physical
object.

`w` relL2 rose with window count (0.260 → 0.356). That may be dilution
(fixed `n_data`, larger pool) or an optimisation trade-off. One control:
`--scale-n-data-with-windows` (`n_data *= n_windows`).

Protocol (do not overwrite `stasis_recon.json`):

```bash
# A. baseline, 3 seeds, 3 cuts  — pair against B
PYTHONPATH=src python3 scripts/public_volume_stasis.py --train \
    --volume data/corrado_lv_phantom/day1_vipr_30min.npz \
    --steps 2000 --n-windows 3 --seeds 0 1 2 --cuts 0.05 0.10 0.15 \
    --out docs/results/public_volume/stasis_recon_w3_seeds.json

# B. |v_beam| oversample, same seeds / windows / cuts
PYTHONPATH=src python3 scripts/public_volume_stasis.py --train \
    --volume data/corrado_lv_phantom/day1_vipr_30min.npz \
    --steps 2000 --n-windows 3 --seeds 0 1 2 --cuts 0.05 0.10 0.15 \
    --fast-frac 0.5 \
    --out docs/results/public_volume/stasis_recon_fastfrac_seeds.json

# C. optional: 1-window, same seeds / cuts (paired window hypothesis)
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

Compare seed-wise: `jet_ml_pred` / `jet_dice` at each cut, and `vel_relL2_w`.
After A–C (D if claiming the window lever in a paper), **freeze Corrado**.
Not a thrombus.
