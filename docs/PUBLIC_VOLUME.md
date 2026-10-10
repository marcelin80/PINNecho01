# Public 4D-flow / phantom volumes (no oracle forcing)

**This is not an IBFE dump.** Public 4D-flow MRI and in-vitro LV phantoms
give a structured velocity volume and (sometimes) a cavity mask. They do
**not** give the IB-spread structural force `f = S[F]`, and 4D-flow MRI does
not give pressure. PINNecho consumes them as :class:`IBFEFrames` with

* `forcing_fluid = 0`
* `traction_wall = 0`
* `pressure_fluid = 0` unless the volume actually carries `p`

so a **baseline vs deliverable-B (`fsi_param`)** comparison can run on a
non-manufactured 3D field. Do **not** train `fsi_informed` on these frames:
the oracle is absent by construction.

> Module: `pinnecho.data.public_volume`
> (`frames_from_public_volume`, `load_public_volume_npz`,
> `synthetic_public_volume_frames`).
> Sweep: `pinnecho-public-ab` / `scripts/public_volume_ab.py`.
> `load_ibfe_output(path)` dispatches here when the NPZ has `velocity` + `mask`.

## Why this exists

Gate R / a real IBAMR export is blocked on the upstream solver. The echo
patient and the CMR/FSI patient are already not the same person, so a
**method** ground truth does not have to come from the in-house cardiac-4D
pipeline. What public data can answer *now* is the deliverable question
(Ablation 8) on a field that is **not** a manufactured NS solution:

> On a real-ish 3D velocity volume, with no access to `f`, does `fsi_param`
> beat kinematic Model A?

That is the clinical input contract (beam-projected Doppler + wall kinematics).
The oracle-forcing question waits for a solver-native `f`.

## On-disk contract

Convert any 4D-flow / phantom release into one NPZ:

| key | shape | units |
|---|---|---|
| `velocity` | `(T, Nx, Ny, Nz, 3)` | m/s |
| `mask` | `(T, Nx, Ny, Nz)` or `(Nx, Ny, Nz)` | cavity interior ≠ 0 |
| `spacing` | `(3,)` | m |
| `origin` | `(3,)` | m (voxel centre of index `(0,0,0)`) |
| `times` | `(T,)` | s |
| `pressure` | `(T, Nx, Ny, Nz)` | Pa, **optional** |
| `_meta` | `[cycle_period, rho, mu, 3]` | s, kg/m³, Pa·s |

Wall points are the mask boundary (6-connected); outward normals are `-∇mask`;
wall velocity is the fluid velocity on those voxels (no-slip proxy).

```bash
# format template (synthetic LV rasterised onto a grid -- not a public dataset)
python scripts/make_example_public_volume.py --out data/public_volume_example/volume.npz

# drop in a real conversion of the same keys, then:
python scripts/public_volume_ab.py --volume path/to/volume.npz \
    --steps 800 --seeds 0 1 2 --out docs/results/public_volume
```

`validate_ibfe_frames` warns that `f` and traction are zero and that this is a
public volume, not an IBFE oracle.

## What the numbers mean

- **Primary metric: per-component / speed velocity relL2.** That is the
  held-out field a 4D-flow volume actually provides.
- **Pressure relL2 is not a metric** when `pressure` is absent (the default).
  The sweep writes `NaN` rather than scoring against a zero field.
- The bundled stand-in (`synthetic_public_volume_frames`) is the analytic 3D
  ellipsoid *rasterised* onto a coarse grid, with `f` and `p` stripped. It is
  for plumbing tests and an example NPZ, **not** a scientific result. A real
  4D-flow or phantom file is what makes the A vs `fsi_param` contrast
  non-manufactured.

## What this does *not* replace

- Resolution gate R (`dx = 0.94 mm`) on the in-house IBAMR run.
- The band-localized **oracle** experiment (Ablation 7 / `--conditions
  oracle`). That needs solver-native `f = S[F]`, which public volumes do not
  have. Reconstructing `f` from `(u, p)` as a momentum residual is the
  manufactured circularity already ruled out; do not do it on 4D-flow MRI.
