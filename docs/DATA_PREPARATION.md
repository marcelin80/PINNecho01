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

### Run the forcing-shuffle diagnostic on real forcing (recommended)

On synthetic data the FSI benefit turned out to be a *trajectory* leak: the
manufactured forcing is derived from the true `u`, so it encodes the answer
(see [`ABLATIONS.md`](ABLATIONS.md) Ablation 6). Real IBFE forcing is
*independently estimated* (structural solver), so the same diagnostic becomes the
decisive test of whether the physics term genuinely helps:

```python
# fsi_informed with the true real forcing vs. a trajectory-decorrelated shuffle
_, m_exact = train_ibfe(frames, backbone="fsi_informed", seed=s)
_, m_shuf  = train_ibfe(frames, backbone="fsi_informed", seed=s, shuffle_forcing=True)
```

Repeat over several seeds and compare vorticity/WSS/pressure. If the exact-minus-
shuffled gap that appears on synthetic data **collapses** for real forcing, the
advantage is a genuine physics prior rather than answer injection — the claim the
synthetic study could not establish.

---

## IBAMR 3D export checklist

Target for the real 3D Doppler-reconstruction pipeline. The whole 3D path is
exercised end-to-end in CI (`tests/test_ibfe_pipeline.py::test_3d_*`,
`tests/test_traction.py::test_*_3d_*`), so matching this contract is sufficient —
no PINNecho code changes are needed. Emit a reference bundle to diff against:

```bash
python scripts/make_example_ibfe_export.py --dim 3 --valve --out data/ibfe_example_3d
```

**Set `spatial_dim = 3`.** Then every array widens by one column vs. 2D:

| array | shape `(N, …)` | columns (exact order for CSV; any order if header-named) |
|---|---|---|
| `coords_fluid`   | `(N_f, 4)` | `x, y, z, t` |
| `velocity_fluid` | `(N_f, 3)` | `u, v, w` |
| `pressure_fluid` | `(N_f, 1)` | `p` |
| `forcing_fluid`  | `(N_f, 3)` | `fx, fy, fz`  (N/m³; structure→fluid body force) |
| `coords_wall`    | `(N_w, 4)` | `x, y, z, t` |
| `normals_wall`   | `(N_w, 3)` | `nx, ny, nz`  (**outward unit**, ‖n‖=1) |
| `velocity_wall`  | `(N_w, 3)` | `uw, vw, ww`  (FSI structural interface velocity) |
| `traction_wall`  | `(N_w, 3)` | `tx, ty, tz`  (`t = σ·n`, Pa) |
| `coords_mitral`  | `(N_m, 4)` | `x, y, z, t`  (optional; residence time) |
| `velocity_mitral`| `(N_m, 3)` | `u, v, w` |
| `coords_aortic` / `velocity_aortic` | `(N_a, 4)` / `(N_a, 3)` | optional outflow |

**3D-specific conventions / pitfalls**
- **Column order is spatial-first then time**: `(x, y, z, t)` — `t` is the *last*
  column, not the third. (The physics operators slice spatial gradients by index
  0..2, so a misplaced `t` silently corrupts `z`-derivatives.)
- **Normals must be outward unit vectors in 3D** (`‖n‖ = 1`); the validator warns
  otherwise. A consistently *inward* normal flips the sign of `traction_wall`.
- **`traction_wall` is the full 3D Cauchy traction** `t_i = -p n_i + Σ_j μ(∂u_i/∂x_j
  + ∂u_j/∂x_i) n_j` (symmetric stress). Export `σ·n` directly from the structural
  stress; do **not** send only the pressure part `-p n`.
- **`forcing_fluid` is the dominant Model-B signal in 3D too** — if it is all
  zeros the FSI backbone collapses to the kinematic baseline (validator warns).
- One bundle = **one cardiac cycle**; keep `t ∈ [0, cycle_period]`.
- Units unchanged (SI): m, s, m/s, Pa, N/m³, kg/m³, Pa·s.

**Verify before training**

```python
import torch; torch.set_default_dtype(torch.float32)
from pinnecho.data import load_ibfe_output, validate_ibfe_frames, train_ibfe
frames = load_ibfe_output("path/to/export_3d")      # .npz, manifest.yaml, or dir
assert frames.spatial_dim == 3
print(validate_ibfe_frames(frames).summary())        # expect OK
# 3D A/B + traction continuity + multi-window Doppler all run unchanged:
model, metrics = train_ibfe(frames, backbone="fsi_informed",
                            use_traction=True, predict_scalar=True)
```

Multi-window 3D Doppler windows are placed automatically with
`default_windows_from_frames(frames, n_windows=3)` (apical + two lateral), or pass
your own transducer apex positions `[(x,y,z), …]` via `transducers=`. For the
realistic **multi-plane** acquisition geometry (each view only insonifies its own
imaging slab), see the next section.

---

## Multi-plane Doppler acquisition geometry

`default_windows_from_frames` / `transducers=` treat each acoustic window as a
point that sees the **whole** fluid volume. A real transthoracic echo exam does
not: it captures a few standard 2D **imaging planes**, and a given view only
resolves the velocity component along its beam **and** only insonifies tissue
lying in a thin slab around its plane. That sparse plane + angular coverage — not
SNR or point count — is the structural bottleneck for 3D reconstruction, so the
measurement layer models it explicitly (`pinnecho.data.acquisition`).

An `ImagingPlane` is a probe `apex` (beam origin) + a unit slab `normal` (+
optional `sector_halfangle`/`max_depth`); `thickness` is the elevation/slice
half-width. The standard set (`VIEWS_3D = a4c, a2c, plax, psax`) is sized to a
cavity's centre/extents (long axis `z`):

| view | probe apex | slab normal | plane |
|---|---|---|---|
| `a4c`  | below (−z) | `+y` | long-axis (x–z) |
| `a2c`  | below (−z) | `+x` | long-axis (y–z), A4C rotated 90° |
| `plax` | side (+x)  | `+y` | parasternal long-axis |
| `psax` | side (+x)  | `+z` | parasternal short-axis (cross-section) |

```python
from pinnecho.data import standard_views_from_frames, train_ibfe
# ~12% of the mean extent ≈ a few-mm slice thickness
planes = standard_views_from_frames(frames, thickness_frac=0.12)   # [a4c, a2c, plax, psax]
model, metrics = train_ibfe(frames, backbone="fsi_informed",
                            use_traction=True, planes=planes)      # planes supersede transducers
```

- Each view contributes only its in-slab points, so fewer / correlated data than
  whole-volume windows — this is the point (realistic observability), not a bug.
- `synthesize_multiplane_doppler(coords, velocity, planes, v_nyquist=, snr_db=,
  sparsity=, dealias=)` is the standalone synth (adds Nyquist aliasing + SNR +
  sparsity); it shares its sparsity/alias/noise tail with `synthesize_doppler`,
  so a fixed seed is reproducible across both.
- **2D** `IBFEFrames` get `VIEWS_2D = apical, lateral` with `normal=None` (an echo
  image *is* 2D, so no slab filter) — the API stays dimension-generic.
- Pick a subset with `standard_views_from_frames(frames, views=("a4c", "plax"))`,
  or build custom `ImagingPlane`s (set `thickness`, `sector_halfangle`, `max_depth`)
  to match a specific probe/protocol.
