# Preparing real IBAMR/IBFE data for PINNecho

This guide explains exactly what to export from your IBAMR/IBFE cardiac FSI run,
in what layout, so that PINNecho can consume it with **no code changes** — and
how each field maps to the three data-gated items:

| Item | Data it needs |
|---|---|
| Swap synthetic → real IBFE | fluid volume + wall interface + FSI body forcing per frame |
| Residence-time reconstruction | a **mitral inflow** point set (to reinit `c = 0`) |
| Higher-fidelity 3D | the same fields with a `z` coordinate + `w` velocity |

Everything is loaded into a single [`IBFEFrames`](../src/pinnecho/data/load_ibfe_output.py)
bundle. You do **not** implement anything: match one of the two on-disk formats
below and call `load_ibfe_output(path)`.

## Quick start

```bash
# 1. Emit a concrete, filled-in example to copy the format from:
python scripts/make_example_ibfe_export.py --dim 2 --valve --out data/ibfe_example
python scripts/make_example_ibfe_export.py --dim 3 --valve --out data/ibfe_example_3d

# 2. Match your export to data/ibfe_example/ (NPZ or manifest+CSV), then:
python - <<'PY'
import torch; torch.set_default_dtype(torch.float32)
from pinnecho.data import load_ibfe_output, validate_ibfe_frames, train_ibfe
frames = load_ibfe_output("path/to/your/export")          # .npz, .yaml, or a dir
print(validate_ibfe_frames(frames).summary())             # sanity-check the contract
model, metrics = train_ibfe(frames, backbone="fsi_informed",
                            use_traction=True, predict_scalar=True)
print(metrics)
PY
```

## Units and conventions

- **SI throughout**: metres, seconds, m/s, Pa, N/m³, kg/m³, Pa·s.
- **Coordinate columns are spatial-first, then time**: 2D `(x, y, t)`, 3D `(x, y, z, t)`.
- One bundle = **one cardiac cycle**, sampled at `T` time frames.
- All arrays are plain point clouds `(N, …)` — no mesh connectivity is required
  (PINNecho is meshless). Sample the fluid volume and the wall as densely as is
  tractable; a few thousand points per frame is plenty for Stage-1-scale runs.

## The fields (what to export and how to compute them)

Per time frame:

**Fluid volume** (Eulerian interior — this is the held-out ground truth *and* the
PDE collocation set):
- `coords_fluid` `(N_f, dim+1)` — interior sample locations + time.
- `velocity_fluid` `(N_f, dim)` — fluid velocity `u` (read from the Eulerian grid,
  interpolated to the sample points).
- `pressure_fluid` `(N_f, 1)` — fluid pressure `p`.
- `forcing_fluid` `(N_f, dim)` — **the term that separates Model B from A.** This is
  the net body force per unit volume the structure exerts on the fluid (active
  contraction + elastic coupling). In IBAMR/IBFE this is the spread Lagrangian
  force density `f = S[F]` on the Eulerian grid (the same `f` added to the fluid
  momentum equation). Export it directly if your solver stores it; otherwise it
  can be recovered as the momentum residual of `(u, p)`.

**Fluid–structure interface** (endocardium):
- `coords_wall` `(N_w, dim+1)`.
- `normals_wall` `(N_w, dim)` — outward unit normals of the interface.
- `velocity_wall` `(N_w, dim)` — the **FSI structural** interface velocity (Model B's
  accurate Dirichlet BC). This differs subtly from the kinematic `d(contour)/dt`
  a segmentation pipeline would give Model A — that difference is the point.
- `traction_wall` `(N_w, dim)` — the structural traction vector `t = σ·n` at the
  interface (Model B's optional traction-continuity target). Interpolate the
  Lagrangian structure stress onto the interface and dot with `n`.

**Valves** (optional; enable residence-time reconstruction):
- `coords_mitral` `(N_m, dim+1)`, `velocity_mitral` `(N_m, dim)` — the mitral inflow
  orifice points + inflow velocity. PINNecho reinitialises residence time `c = 0`
  here at every time step, so this only needs to cover the inflow boundary.
- `coords_aortic`, `velocity_aortic` — the aortic outflow orifice (optional).

## Format 1 — NPZ bundle (simplest)

A single `.npz` with these keys (see `pinnecho/data/ibfe_io.py::NPZ_TENSOR_KEYS`):

```
coords_fluid, velocity_fluid, pressure_fluid, forcing_fluid,
coords_wall, normals_wall, velocity_wall, traction_wall,
coords_mitral, velocity_mitral, coords_aortic, velocity_aortic,
_meta   # = [cycle_period, rho, mu, spatial_dim]
```

Write it with `save_ibfe_npz(frames, "bundle.npz")`, or build the arrays yourself
and `np.savez` with exactly those keys. Load with `load_ibfe_output("bundle.npz")`.

## Format 2 — manifest + per-frame CSV (for frame-by-frame exports)

A `manifest.yaml` plus one CSV per field per frame. Columns are matched by
**header name** (order-independent, extra columns ignored). See the annotated
[`configs/ibfe_manifest.template.yaml`](../configs/ibfe_manifest.template.yaml).

```yaml
spatial_dim: 2
cycle_period: 0.9
rho: 1060.0
mu: 0.0035
frames:
  - t: 0.00
    fluid:  frames/fluid_000.csv    # cols: x,y[,z], u,v[,w], p, fx,fy[,fz]
    wall:   frames/wall_000.csv      # cols: x,y[,z], nx,ny[,nz], uw,vw[,ww], tx,ty[,tz]
    mitral: frames/mitral_000.csv    # cols: x,y[,z], u,v[,w]   (optional)
  - t: 0.11
    ...
```

For 3D set `spatial_dim: 3` and add the `z` / `w` / `nz` / `ww` / `tz` / `fz`
columns. A VTK/Exodus hook (`load_frame_vtk`, needs `meshio`) is available if you
prefer to point frames at `.vtu`/`.vtk` files with a field-name mapping.

## Validation checklist

Run `validate_ibfe_frames(frames)` and confirm `OK`. It checks:

- array shapes and per-group row-count consistency,
- finiteness (no NaN/Inf),
- wall normals are unit length,
- times lie within one cycle (use `load_ibfe_output(..., time_range=(t0, t1))`
  to restrict, and `subsample=N` to thin very dense meshes),
- physical-magnitude sanity (speed in m/s, non-zero forcing, `rho, mu > 0`).

Full PDE self-consistency (divergence-free, NS-exact forcing) is a property of a
*differentiable field*, so it is guaranteed by the synthetic generator and its
unit tests rather than checkable from scattered real samples — but the validator
will catch the mistakes that actually break training (wrong shapes, unit errors,
missing forcing, non-unit normals).

## From `IBFEFrames` to a trained model

- `train_ibfe(frames, backbone=..., use_traction=..., predict_scalar=...)` wires the
  bundle to the shared network + composite loss (Doppler data on the beam
  component only, PDE + FSI forcing at collocation, wall/valve Dirichlet, optional
  traction continuity, residence-time inflow reinit) and runs Adam + L-BFGS.
- `evaluate_ibfe(model, frames)` reports held-out velocity/pressure error vs the
  fluid ground truth.
- For A-vs-B, call it twice with `backbone="baseline"` and `"fsi_informed"` on the
  same `frames` (identical architecture/data/init) — the same clean ablation used
  on the synthetic case.
