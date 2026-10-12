# Corrado — 3-window baseline, 3 seeds, 3 cuts

A protocol. Uniform fluid sampling. `data_fast_frac=None` (`|v_beam|` not
used). 2000 Adam + 50 L-BFGS, 3 windows, seeds 0/1/2. Same
`day1_vipr_30min.npz`. Not a thrombus. Do not rerun.

Laptop JSON: `stasis_recon_w3_seeds.json`. A did **not** write
`*_peaks.npz`; do not rerun A to get them. Peak-ratio and blur-GT are
no-retrain checks on B / a blurred GT.

Jet = above the cut for **> 50 % of the cycle** (`cavity & ~persist`),
not “ever exceeded.”

GT remainder (cavity − persist) depends on the cut:

| cut | GT jet |
|---|---|
| 5 cm/s | 79.74 mL |
| 10 cm/s | 30.38 mL |
| 15 cm/s | 10.80 mL |

| seed | speed relL2 | 5 cm/s pred / Dice | 10 cm/s pred / Dice | 15 cm/s pred / Dice |
|---|---|---|---|---|
| 0 | 0.417 | 72.00 / 0.831 | 26.23 / 0.806 | 7.97 / 0.625 |
| 1 | 0.461 | 74.50 / 0.840 | 24.17 / 0.796 | 5.28 / 0.532 |
| 2 | 0.414 | 73.11 / 0.844 | 25.39 / 0.818 | 5.58 / 0.525 |
| mean | 0.431 | 73.2 / 0.838 | **25.3 / 0.807** | **6.3 / 0.561** |

Signed relative jet-volume error `(pred − GT) / GT`:

| cut | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 5 cm/s | −10 % | −7 % | −8 % |
| 10 cm/s | −14 % | −20 % | −16 % |
| 15 cm/s | −26 % | −51 % | −48 % |

Seed 0 at 10 cm/s matches the earlier one-seed row (26.2 mL, Dice ≈ 0.80).
Across seeds the 10 cm/s jet is **24.2–26.2 mL** (always a 4–6 mL leak).
That leak is not a one-seed accident.

The 15 cm/s core (10.8 mL GT, ~1350 voxels at 2 mm) is the quantity that
actually fails: Dice 0.53–0.63, and seeds 1–2 keep only ~5.4 mL. The
5 cm/s remainder looks better (Dice ~0.84) because it is a large,
slow-ish set — the same majority-class effect as persist Dice. Do not
quote 5 cm/s as “the jet is recovered.”

Error growing with the cut is compatible with a constant peak-speed
underestimate (tail clips first). That ratio is not in this file; B
saves `*_peaks.npz` so it can be checked without a retrain.

The 1-window 3-seed pair is
[`STASIS_RECON_W1.md`](STASIS_RECON_W1.md). 10 cm/s 23.4–24.0 mL vs
this table 24.2–26.2 mL. B is
[`STASIS_RECON_FASTFRAC.md`](STASIS_RECON_FASTFRAC.md) (rejected).
Do not rerun A, B, or C.
