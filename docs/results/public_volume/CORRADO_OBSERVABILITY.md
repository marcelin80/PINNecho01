# Corrado LV phantom — window-count observability

First non-manufactured check of the window-count lever on an **LV-cavity**
4D-flow field. Wisconsin in-vitro phantom (Corrado et al., *Magn Reson Med*
2021; OSF [2naw4](https://osf.io/2naw4/)), Day 1 / PC VIPR / 30 min, converted
by `scripts/convert_corrado_lv.py`. Backbone `baseline` only. `f = 0`, no
pressure GT. Not an IBFE oracle. Not a thrombus study.

Windows (from `default_windows_from_frames`): apical (`−z`, observes `w`),
lateral-x (observes `u`), lateral-y (observes `v`).

## 2000 Adam + 50 L-BFGS, 3 seeds (primary)

| condition | speed | u | v | w | p |
|---|---|---|---|---|---|
| windows=1 | 0.506 | 0.728 | 0.898 | **0.260** | nan |
| windows=2 | 0.432 | **0.409** | 0.834 | 0.339 | nan |
| windows=3 | **0.418** | 0.423 | **0.690** | 0.356 | nan |

Per-seed `u`: 1-window 0.757 / 0.674 / 0.753; 2-window 0.415 / 0.399 / 0.413.
The 1→2 drop in `u` and the 1→3 drop in `v` sit well outside seed scatter.
The already-observed apical `w` *rises* (0.260 → 0.356) as windows are added —
the same trade-off as the manufactured 2D sweep. Speed improves 0.506 → 0.418.

Raw JSON on the laptop: `docs/results/public_volume/public_volume_observability.json`
(overwrites; 400-step single-seed backup is
`public_volume_observability_steps400_seed0.json`).

## 400 Adam + 50 L-BFGS, seed 0 (under-trained control)

| condition | speed | u | v | w |
|---|---|---|---|---|
| windows=1 | 0.691 | 0.896 | 1.008 | 0.520 |
| windows=2 | 0.675 | 0.694 | 0.964 | 0.610 |
| windows=3 | 0.676 | 0.724 | 0.900 | 0.614 |

Same *direction*, washed-out magnitudes. Speed is flat. Do not quote this table
as the result; it is why the 2000-step / 3-seed run exists.

## What this does *not* say

- The field is recovered (`speed` relL2 0.42, 3-window `v` 0.69).
- FSI helps (no `f`; baseline only).
- Stasis / residence time / thrombus maps are trustworthy at this error.
  GT stasis on the same volume is 165 / 195 mL persistent at 10 cm/s
  ([`STASIS_GT.md`](STASIS_GT.md)).
