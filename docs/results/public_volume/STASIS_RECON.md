# Corrado — reconstruction vs GT stasis (10 cm/s)

Same cut as [`STASIS_GT.md`](STASIS_GT.md). Baseline, 2000 Adam + 50 L-BFGS,
seed 0, `day1_vipr_30min.npz`. Not a thrombus. Speed relL2 matches the
observability seed-0 rows (1-window 0.516 vs ~0.51; 3-window 0.417 vs 0.418).

| | GT | 1 window | 3 windows |
|---|---|---|---|
| persist stasis | 164.6 mL | 171.3 mL | 168.8 mL |
| **jet** | **30.4 mL** | **23.7 mL (−22 %)** | **26.2 mL (−14 %)** |
| persist Dice | | 0.959 | 0.967 |
| **jet Dice** | | **0.748** | **≈ 0.80** |
| stasis frac | 0.831 | 0.856 | 0.851 |
| speed mean | 6.16 cm/s | 5.70 cm/s | 5.68 cm/s |
| vel_relL2_speed | | 0.516 | 0.417 |

Extra windows recover **2.5 mL** of the jet on seed 0. The 3-seed pair
is now in [`STASIS_RECON_W1.md`](STASIS_RECON_W1.md): 1-window 10 cm/s
23.4–24.0 mL vs A 24.2–26.2 mL (mean +1.6 mL). 15 cm/s is not uniformly
better with 3 windows. Persist Dice stays ~0.96 — do not quote it.

This is **not** enough for residence-time or thrombus maps. Registered
PIV does not referee that 30 mL ([`STASIS_PIV.md`](STASIS_PIV.md)).
Do not rerun these one-seed files.

Laptop JSON: `stasis_recon_w1.json` (1 window), `stasis_recon.json` (3 windows).
3-window `jet_dice` is derived from the same voxel-volume identity as
`compare_volume_stasis`; the 1-window file has it natively.
