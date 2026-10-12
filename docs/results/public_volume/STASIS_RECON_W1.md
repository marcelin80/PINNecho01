# Corrado — C, 1-window × 3 seeds

Pre-registered in [`JET_PRESERVE.md`](JET_PRESERVE.md). 2000 Adam + 50
L-BFGS, **1 window**, seeds 0/1/2, `n_data=3072` fixed (same total
Doppler rows as A). No `--fast-frac`. Same `day1_vipr_30min.npz`. Not a
thrombus. Do not rerun.

Laptop JSON: `stasis_recon_w1_seeds.json`.
Peaks: `stasis_recon_w1_seeds_seed{0,1,2}_peaks.npz`.

Log stamp: `windows=1 n_data=3072 data_fast_frac=None source=none`.
Seed 0 at 10 cm/s is 23.70 mL / Dice 0.748 — matches the earlier
one-seed 1-window row.

## Decision at 10 cm/s

1-window signed error had to be worse than A on **all three** seeds
despite ~3× the per-window points. Then extra windows help even under
dilution.

| seed | C 10 cm/s signed | A 10 cm/s signed |
|---|---|---|
| 0 | **−22.0 %** | −14 % |
| 1 | **−20.9 %** | −20 % (near-tie) |
| 2 | **−22.9 %** | −16 % |

| seed | C 10 cm/s mL / Dice | A 10 cm/s mL / Dice |
|---|---|---|
| 0 | 23.70 / 0.748 | 26.23 / 0.806 |
| 1 | 24.02 / 0.782 | 24.17 / 0.796 |
| 2 | 23.42 / 0.764 | 25.39 / 0.818 |
| mean | **23.7 / 0.765** | **25.3 / 0.807** |

3-window A is better on every seed. Seed 1 is a near-tie in millilitres
(24.02 vs 24.17). Precision on C is 0.85–0.89 — not a fake jet, just
lower recall (0.67–0.70). The one-seed +2.5 mL is now **+1.6 mL mean**
across seeds, under a confound that *favours* 1-window.

That is enough to drop the “one-seed hypothesis” label. It is **not**
enough to title a paper “windows protect the jet.” The effect is modest.
`n_data` is still fixed, so D is required only if that lever is a paper
claim. Do not run D otherwise.

## 15 cm/s core — do not claim windows save it

| seed | C 15 cm/s | A 15 cm/s |
|---|---|---|
| 0 | 4.34 (−60 %) Dice 0.467 | 7.97 (−26 %) 0.625 |
| 1 | 4.62 (−57 %) Dice 0.492 | 5.28 (−51 %) 0.532 |
| 2 | 6.79 (−37 %) Dice 0.565 | 5.58 (−48 %) 0.525 |

Seeds 0–1 are worse on 1-window. Seed 2 volume/Dice is *better* on
1-window. The 10.8 mL core still fails either way. Peak-ratio median
0.76–0.82 at 15 cm/s (C) vs ~0.80–0.84 (B). Same amplitude-bias story.

## Speed

C `vel_relL2_speed` 0.516 / 0.499 / 0.460 (mean 0.492). A: 0.417 /
0.461 / 0.414 (mean 0.431). Matches the observability 1- vs 3-window
speed gap. Slow-region `slow_speed_relL2` at 10 cm/s is 0.408 / 0.390 /
0.368 — not a collapse.

Do not quote 5 cm/s (C Dice 0.82–0.84, the large remainder).

After C: freeze Corrado unless D is needed for a window claim. Not a
thrombus.
