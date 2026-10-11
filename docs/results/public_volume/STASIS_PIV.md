# Corrado — MRI vs registered PIV (10 cm/s, overlap)

Same 71×53×87×20 grid, `origin_delta_vox = 0`. PIV mask sits entirely
inside the MRI cavity (73 / 195 mL = 37 %). Fixed clinical cut. No PINN.
Not a thrombus.

| | MRI full | PIV full | overlap (73 mL) |
|---|---|---|---|
| jet at 10 cm/s | **30.4 mL** | 0.016 mL | MRI **0.0** / PIV 0.016 |
| persist Dice | | | 0.9999 |
| jet Dice | | | 0.0 |
| speed mean | 6.16 cm/s | 3.95 cm/s | MRI 3.75 / PIV 3.95 |
| speed p99 | 26.1 cm/s | 9.5 cm/s | |
| speed relL2 | | | 0.487 |

The MRI 30.4 mL jet is **entirely outside the PIV FOV**. On the overlap both
fields are the stasis sea (persist Dice 0.9999). PIV `p99` is 9.5 cm/s, so
almost no PIV voxel crosses 10 cm/s (two voxels, 0.016 mL). Speed means on
the overlap agree at ~4 cm/s; the vector field still differs (relL2 0.49).

PIV cannot referee the PINN jet (23.7 / 26.2 vs 30.4 mL). It never sees that
jet. The auto-cut PIV row (2.4 cm/s, “jet” 55.6 mL) is not a comparison —
different mask, different cut. Do not quote it.

Laptop JSON: `docs/results/public_volume/stasis_mri_vs_piv.json`.
Do not rerun.
