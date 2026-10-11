# Corrado — how to keep the 30 mL jet (sampling lever)

The 3-window PINN still loses 4 mL of jet (30.4 → 26.2 mL) at speed relL2
0.42. That is not an IBFE problem. The cavity is 84 % persistent stasis, and
`frames_from_public_volume` drew **uniform** fluid points (`n_fluid=2000` /
20 frames ≈ 100 / frame → ~16 jet points). The beam data loss is a global
mean, so the stasis sea dominates the count.

End-of-train 3-window weights: `10*data + 1*pde + 10*bc` ≈ 1.37 + 0.19 +
1.82. PDE is ~6 % of the total. Down-weighting PDE is the second knob, not
the first. Windows already bought +2.5 mL (1 → 3). Change **one** lever.

`--fast-frac 0.5` reserves half of each frame's fluid samples for voxels
with speed ≥ 10 cm/s. Same 2000 steps, 3 windows, seed 0. Look at
`jet_ml_pred` and `jet_dice` vs 26.2 mL / ≈0.80. `sample_fast_frac` on
the console should be ~0.5, not ~0.16.

```bash
PYTHONPATH=src python3 scripts/public_volume_stasis.py --train \
    --volume data/corrado_lv_phantom/day1_vipr_30min.npz \
    --steps 2000 --n-windows 3 --seed 0 --fast-frac 0.5 \
    --out docs/results/public_volume/stasis_recon_fastfrac.json
```

Do not combine with a PDE-weight change. Do not overwrite
`stasis_recon.json`. Not a thrombus.
