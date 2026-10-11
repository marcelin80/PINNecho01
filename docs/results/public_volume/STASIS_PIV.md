# Corrado — MRI vs registered PIV (do not quote the auto-cut row)

PIV (`PIV_Data/Registered to Day 1/PC VIPR/Average Beat`) is on the **same
71×53×87×20 grid** as Day 1 PC VIPR 30 min, but it is **not** the same
cavity. The naive GT dump used the auto-shrink heuristic (`p99 ≈ 9.5 cm/s
< 10 cm/s` → cut 2.4 cm/s) on the PIV mask alone:

| | MRI (10 cm/s) | PIV auto (2.4 cm/s) |
|---|---|---|
| cavity | 195.0 mL | **73.0 mL** |
| persist | 164.6 mL | 17.4 mL |
| jet | 30.4 mL | 55.6 mL |
| speed mean | 6.16 cm/s | 3.95 cm/s |
| speed p99 | 26.1 cm/s | 9.5 cm/s |

Do **not** compare 30.4 vs 55.6. Different mask, different cut. Units look
fine (`mm/s` → m/s; PIV `speed_max` 15 cm/s is physically plausible). The
PIV FOV is ~37 % of the MRI cavity.

The comparison that matters is the **overlap mask at a fixed 10 cm/s cut**.
That command is `--volume` MRI `--other` PIV (no `--train`):

```bash
PYTHONPATH=src python3 scripts/public_volume_stasis.py \
    --volume data/corrado_lv_phantom/day1_vipr_30min.npz \
    --other data/corrado_lv_phantom/day1_piv_avg.npz \
    --speed-thresh 0.10 \
    --out docs/results/public_volume/stasis_mri_vs_piv.json
```

Look at `overlap_ml`, `overlap_jet_ml_true` (MRI), `overlap_jet_ml_pred`
(PIV), `overlap_jet_dice`, `overlap_speed_relL2`, `origin_delta_vox`.
If `piv_jet_ml` at 10 cm/s is ~0, PIV cannot police the MRI 30 mL jet:
almost no PIV voxel reaches that cut.

Not a thrombus. Not a PINN result.
