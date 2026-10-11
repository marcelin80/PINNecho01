# Corrado — echo-plane slab coverage of the 30 mL jet

No training. Not a thrombus. Same 10 cm/s jet as
[`STASIS_GT.md`](STASIS_GT.md) (30.38 mL). Views are AABB-canonical
A4C/A2C/PLAX/PSAX (long axis `z`), same as `plane_coverage`, **not** a
cardiologist on the phantom. Half-thickness `0.12 ×` mean cavity AABB.

| protocol | jet in slab | frac | along-beam | cavity in slab |
|---|---|---|---|---|
| a4c | **22.31 / 30.38 mL** | **0.734** | 0.847 | 121.6 / 195 mL (0.62) |
| a4c+a2c | 24.57 | 0.809 | 0.825 | 159.9 (0.82) |
| a4c+plax | 22.31 | 0.734 | 0.892 | 121.6 (0.62) |
| 4-view | 24.70 | 0.813 | 0.865 | 169.9 (0.87) |

A4C already hits 22 mL of the 30 mL jet. A2C adds 2.3 mL. PLAX equals A4C
because they share slab normal `+y` (same midplane, different apex). PSAX
adds 0.13 mL. About 5.7 mL of jet sits off every midplane.

That is **not** “echo sees the CHF marker.” These slabs are fat: A4C
insonifies 62 % of the whole cavity, and the jet is only mildly enriched
on that midplane (73 % vs 62 %). Along-beam ~0.85 says that *covered* jet
voxels are mostly projected; the leftover 19 % is a coverage hole, not a
beam-angle hole.

Laptop JSON: `docs/results/public_volume/slab_coverage.json`. Do not rerun
at `thickness_frac=0.12`. A thinner cut (`--thickness-frac` ~0.04, ~5 mm
half-thickness) is the only follow-up if you want a clinical-slice bound.
