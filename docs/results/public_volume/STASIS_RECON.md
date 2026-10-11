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

Extra windows recover **2.5 mL** of the jet **on seed 0 only**. The
3-window leak itself is now 3-seed
([`STASIS_RECON_SEEDS.md`](STASIS_RECON_SEEDS.md)): 10 cm/s jet
24.2–26.2 mL (Dice ~0.81); 15 cm/s core Dice ~0.56. “Windows protect the
jet” is still a hypothesis until 1-window is paired on the same seeds.
Persist Dice stays ~0.96 — do not quote it.

This is **not** enough for residence-time or thrombus maps. Registered
PIV does not referee that 30 mL ([`STASIS_PIV.md`](STASIS_PIV.md)).
Do not rerun these one-seed files.

Laptop JSON: `stasis_recon_w1.json` (1 window), `stasis_recon.json` (3 windows).
3-window `jet_dice` is derived from the same voxel-volume identity as
`compare_volume_stasis`; the 1-window file has it natively.
