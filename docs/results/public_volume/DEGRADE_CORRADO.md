# Corrado + our synthesizer — degradation fallback (pre-registered)

DUPLEX public files have no CFD vector GT
([`DUPLEX.md`](../../data/DUPLEX.md)). Contact is off the essential
path. Robustness, if claimed at all, is **our** sparsity / SNR / wrap
applied to the **measured** Corrado field. That is less realistic than
SIMUS. Say so in the paper.

Do **not** rerun A/B/C. Do not move these rules after seeing millilitres.

## Why clinical aliasing is probably inactive here

Locked Corrado GT ([`STASIS_GT.md`](STASIS_GT.md)): `speed_p99` =
**0.261 m/s**, mean 0.062 m/s. DUPLEX Nyquist is **1.078 m/s**.
A sample that never exceeds p99 cannot wrap at that rail.

Gate (code: `pinnecho.eval.degrade`): `wrap_frac` on jet beam samples
must be ≥ **0.01** for `alias_clinical` to be an active step. If not,
drop that row. Do not invent bedside aliasing from this phantom.

`alias_ours_15cms` (Nyquist 0.15 m/s) is **ours**. Use it only if the
text says the wrap is not a clinical PRF.

## Ladder (locked)

| id | `v_nyquist` | SNR | sparsity | dealias | role |
|---|---|---|---|---|---|
| `ideal` | 10 m/s | 40 dB | 0 | yes | baseline; closest to A/B/C projection |
| `snr10` | 10 | 10 dB | 0 | yes | noise |
| `sparse50` | 10 | 40 dB | 0.50 | yes | acquisition gaps |
| `alias_clinical` | 1.078 | 40 dB | 0 | no | DUPLEX-matched rail; likely inactive |
| `alias_ours_15cms` | 0.15 | 40 dB | 0 | no | labelled-ours wrap |

A/B/C used beam projection plus `noise_level = 0.05`, not this ladder.
`ideal` is the new baseline for **deltas**, not a replacement A.

## Success criteria (before any new PINN number)

1. Write `ideal` first. Main numbers are **Δ vs ideal** on the same
   metrics as Corrado A/B: 10 / 15 cm/s jet mL and signed error,
   `peak_speed_report` median on the GT jet, speed relL2.
2. The frozen B peak-ratio table (**~0.80–0.88**, median ~0.83 on the
   10–15 cm/s jet) is the amplitude reference. Same function
   (`peak_speed_report`). Do not rerun B to “reproduce” it.
3. If `ideal` is already as bad as A/B on 15 cm/s (Dice ~0.56 / peak
   ~0.83), a further drop under `snr10` / `sparse50` is a limitation,
   not a robustness title.
4. If `alias_clinical` wrap fraction < 0.01, omit it from claims.
5. Do not quote persist Dice, 5 cm/s Dice, or a cohort.
6. One seed is enough to see the direction; three seeds only if `ideal`
   is worth the cost. Do not start until the paper draft needs the
   paragraph.

## Peak ratio 0.83 (already recorded)

B, GT jet, `max_t |u|` pred/GT median
([`STASIS_RECON_FASTFRAC.md`](STASIS_RECON_FASTFRAC.md)):

| cut | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 5 cm/s | 0.888 | 0.871 | 0.834 |
| 10 cm/s | 0.879 | 0.850 | 0.821 |
| 15 cm/s | 0.838 | 0.835 | 0.798 |

Quote the range **0.80–0.88** or “~0.83”, not a new run. Blur-GT peak
0.88–0.90 ([`STASIS_BLUR_GT.md`](STASIS_BLUR_GT.md)).
