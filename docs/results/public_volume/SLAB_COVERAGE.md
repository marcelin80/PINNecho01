# Corrado — echo-plane slab coverage of the 30 mL jet

No training. Not a thrombus. Same 10 cm/s jet as
[`STASIS_GT.md`](STASIS_GT.md). Views are the project's AABB-canonical
A4C/A2C/PLAX/PSAX (`standard_views_from_extents`, long axis `z`), **not** a
cardiologist placing probes on the silicone phantom. Slab half-thickness is
`0.12 ×` mean cavity extent (same as `plane_coverage`).

```bash
PYTHONPATH=src python3 scripts/public_volume_slab.py \
    --volume data/corrado_lv_phantom/day1_vipr_30min.npz \
    --out docs/results/public_volume/slab_coverage.json
```

| column | meaning |
|---|---|
| `jet_ml` | 10 cm/s through-flow remainder (~30 mL on MRI) |
| `jet_ml_in_slab` / `jet_frac_in_slab` | how much of that jet sits in the protocol |
| `jet_along_beam_frac` | on covered jet voxels, mean max_view \|v̄·b\| / \|v̄\| |
| `cavity_ml_in_slab` | cavity volume the slabs insonify |

If A4C already covers most of the 30 mL, a realistic apical view can see the
marker. If even 4-view leaves most of it out, the jet is a 4D-flow object and
the echo→CHF story stops here.

Do not train. Do not quote until the laptop JSON exists.
