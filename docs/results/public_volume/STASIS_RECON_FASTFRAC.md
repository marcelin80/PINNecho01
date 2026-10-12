# Corrado — B, `|v_beam|` per-frame p80 oversample

Pre-registered in [`JET_PRESERVE.md`](JET_PRESERVE.md). 2000 Adam + 50
L-BFGS, 3 windows, seeds 0/1/2, `n_data=3072` fixed,
`--fast-frac 0.5`, fast set = per-frame `|v_beam|` p80 (not 5/10/15 cm/s,
not the GT jet). Same `day1_vipr_30min.npz`. Not a thrombus. Do not rerun.

Laptop JSON: `stasis_recon_fastfrac_seeds.json`.
Peaks: `stasis_recon_fastfrac_seeds_seed{0,1,2}_peaks.npz`.

Log stamp: `source=per_frame_vbeam_quantile quantile=0.8 thresh=None`.
The earlier absolute `|v_beam| ≥ 0.10` scout is not this file.

## Decision: reject the sampling lever

15 cm/s signed jet error had to be ≥ −25 % on **all three** seeds.

| seed | 15 cm/s signed | bar −25 % |
|---|---|---|
| 0 | **−23.3 %** | pass |
| 1 | **−31.9 %** | fail |
| 2 | **−26.1 %** | fail |

Slow region and precision did **not** fail the other two clauses:
`vel_relL2_speed` 0.403 / 0.448 / 0.400 (A: 0.417 / 0.461 / 0.414);
10 cm/s precision 0.837 / 0.857 / 0.873; 10 cm/s `slow_speed_relL2`
0.397 / 0.383 / 0.362 (each ≤ that seed’s A speed + 0.03).

The lever is dropped because the core bar failed, not because the slow
region was sacrificed. Do not move the bar after seeing these numbers.

## Volume / Dice vs A

| seed | speed | 5 cm/s pred / Dice / err | 10 cm/s pred / Dice / err | 15 cm/s pred / Dice / err |
|---|---|---|---|---|
| 0 | 0.403 | 82.09 / 0.844 / +2.9 % | 29.01 / 0.817 / −4.5 % | 8.28 / 0.643 / −23.3 % |
| 1 | 0.448 | 83.03 / 0.848 / +4.1 % | 29.14 / 0.839 / −4.1 % | 7.36 / 0.685 / −31.9 % |
| 2 | 0.400 | 76.70 / 0.867 / −3.8 % | 27.58 / 0.831 / −9.2 % | 7.98 / 0.646 / −26.1 % |
| mean | 0.417 | 80.6 / 0.853 | **28.6 / 0.829** | **7.9 / 0.658** |

A at the same cuts: 10 cm/s 25.3 mL (−14/−20/−16 %), 15 cm/s 6.3 mL
(−26/−51/−48 %), Dice ~0.56. B recovered most of the 10 cm/s remainder
(leak 4–6 mL → 1.4 / 1.2 / 2.8 mL) with high precision. That is not
fake-jet millilitres. It is also not the adopt criterion.

15 cm/s recall stays ~0.56–0.58: the core voxels are still half-missing.
Dice 0.64–0.69 is better than A’s 0.53–0.63 and still a failure of the
10.8 mL core. Do not quote 5 cm/s (slight overshoot on seeds 0–1,
precision ~0.83).

## Peak ratio: amplitude bias

Pred / GT `max_t |u|` median on the GT jet (printed `peak=`):

| cut | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 5 cm/s | 0.888 | 0.871 | 0.834 |
| 10 cm/s | 0.879 | 0.850 | 0.821 |
| 15 cm/s | 0.838 | 0.835 | 0.798 |

The ratio sits at **~0.80–0.88** on every seed, a little lower on the
15 cm/s core. A constant peak under-estimate clips the tail first. That
is the result. Sampling moved the 10 cm/s boundary; it did not restore
the high-speed core.

## What this does not close

Blur-GT (no train) is still the resolution ceiling for 15 cm/s Dice
~0.66. C (`--n-windows 1`) is still required before claiming that
windows protect the jet. `|v_beam|` cannot see a cross-beam jet.

Not a thrombus.
