# Corrado LV phantom — GT Eulerian stasis (10 cm/s)

Kinematic marker on the **measured** 4D-flow field
(`data/corrado_lv_phantom/day1_vipr_30min.npz`). No PINN, no `f`, no
coagulation model. Day 1 / PC VIPR / 30 min.

Clinical cut: cavity voxel speed `< 0.10 m/s`. Persistent = below that cut
for ≥ 50 % of the 20-frame cycle.

| quantity | value |
|---|---|
| `speed_thresh` | **0.10 m/s** (10 cm/s) |
| `speed_mean` | 0.062 m/s |
| `speed_p99` | 0.261 m/s |
| `cavity_ml` | 195.0 mL |
| `stasis_frac` (voxel × time) | 0.831 |
| `persist_stasis_frac` | 0.844 |
| `persist_stasis_ml` | **164.6 mL** |
| moving remainder | ~30 mL |

The first pass used an automatic cut of 6.5 cm/s (`0.25 × p99`) and is
superseded. Do not quote 140 mL / 0.70.

This rigid through-flow phantom is already stasis-rich: most of the 195 mL
cavity sits below 10 cm/s. The structure that a reconstruction must keep is
the ~30 mL jet, not a healthy emptying LV. A PINN with speed relL2 ~ 0.42
is expected to smear that jet and inflate persistent stasis toward the full
cavity. Score that with one 2000-step / 3-window reconstruction (do not
overwrite this GT JSON):

```bash
python scripts/public_volume_stasis.py --train \
    --volume data/corrado_lv_phantom/day1_vipr_30min.npz \
    --steps 2000 --n-windows 3 --seed 0 \
    --out docs/results/public_volume/stasis_recon.json
```

Look at `jet_ml_true` (~30 mL) vs `jet_ml_pred` and `jet_dice`, not only
`persist_stasis_dice`. 1- vs 3-window, seed 0:
[`STASIS_RECON.md`](STASIS_RECON.md). It is not a thrombus.
