# Corrado — reconstruction vs GT stasis (10 cm/s)

One baseline run, 2000 Adam + 50 L-BFGS, 3 windows, seed 0, on
`day1_vipr_30min.npz`. Same clinical cut as [`STASIS_GT.md`](STASIS_GT.md).
Not a thrombus. `vel_relL2_speed = 0.417` matches the observability
windows=3 / seed=0 speed error.

| quantity | GT | PINN |
|---|---|---|
| persist stasis | 164.6 mL | 168.8 mL |
| jet (cavity − persist) | **30.4 mL** | **26.2 mL** |
| stasis frac (voxel × time) | 0.831 | 0.851 |
| speed mean | 6.16 cm/s | 5.68 cm/s |
| persist Dice | 0.967 | |
| jet Dice (derived) | **≈ 0.80** | |

`persist_stasis_dice = 0.967` is the majority class: 84 % of the cavity is
already slow, so a slightly-too-slow reconstruction still looks like the
stasis sea. The quantity that matters for the CHF motivation is the jet:
it shrinks 30.4 → 26.2 mL (−14 %) and the moving-region Dice is ~0.80, not
0.97.

That is compatible with speed relL2 0.42 (smoothing). It does **not** say
the reconstructed field is good enough for residence-time or thrombus maps.
It says the 10 cm/s persist map on this phantom is a coarse label that
mostly survives, while the 30 mL through-flow core is already leaking.

Do not rerun unless the conversion or the windows change.
`docs/results/public_volume/stasis_recon.json` on the laptop is the raw file.
