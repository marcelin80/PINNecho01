# Decision 9 — unstressed reference (locked)

Locked 2026-10-13. This is an **essential-path** item, not something
to postpone until a third retune. PINNecho's core interest is
diastolic filling (E-wave, IVPG, vortex ring). Those phenomena exist
only if the ventricle actually fills by tens of millilitres.

Code: `pinnecho.data.ibfe_fidelity`. On-disk name: **`/meta/fidelity`**
(NPZ key `_fidelity`, manifest key `fidelity`).

## Locked choice

| option | meaning | status |
|---|---|---|
| **(a)** | Treat the image as already loaded. Build an unstressed reference V0, then fill toward the image volume with atrial pressure. | **LOCKED** |
| (b) | Use the image volume as the unstressed reference. | Rejected. Almost no fill. No E-wave. Cannot test the current PINNecho hypotheses. |
| (c) | Record the missing fill as a limitation and move on. | Rejected. Same as (b): it abandons the purpose. |

## Two-stage (a)

1. **Approximate first (now).** Isotropically scale the image mesh to
   V0, then fill with atrial pressure toward the image volume.
   Implementation is simple. **This is an approximation:** wall
   thickness scales with the mesh (about 7.4 mm → about 6 mm) and the
   strain state is not the in-vivo loaded state. Say so in every
   export that uses it (`decision_9: approx`, `approximation: true`).
2. **Formal later (not in PINNecho).** Backward displacement: the
   unstressed shape that deforms to the image geometry under
   endocardial pressure `p_img`. Each IBFE iteration is a static
   solve. Tag only: `decision_9: formal`. PINNecho does not run the
   inverse.

Do **not** implement the formal inverse in this repository.

## Exposed assumed parameters

`p_img` is **unknown**. Phase is not locked. Assume mid–late diastole
**5–10 mmHg** and check sensitivity at both ends.

Klotz V0 ≈ **58 mL** is a single-beat empiric with large error.
pat12 is congenital; compliance may differ. Image volume is about
**105 mL**. The linear isotropic scale is

```
(V0 / V_img)^{1/3} = (58 / 105)^{1/3} ≈ 0.82
```

V0 sets the fill volume, so this ratio is an **exposed assumed
parameter**, not a measurement. Helpers:

- `isotropic_linear_scale(v0_ml, image_volume_ml)`
- `decision_9_approx_fidelity(...)` (writes `v0_scale`, `v0_ml`,
  `p_img_mmhg`)

Do not hide 0.82 inside a mesh script.

## `/meta/fidelity`

| field | pre-D9 image-IBFE (default if missing) | D9 approx | manufactured / public volume |
|---|---|---|---|
| `source` | `image_ibfe` | `image_ibfe` | `manufactured` / `public_volume` |
| `decision_9` | `not_applied` | `approx` | `not_applicable` |
| `diastole_usable` | **false** | true (approximation) | true |
| `usable_phases` | `systole`, `ejection` | + `diastole` | `all` |
| `reference` | `image_as_unstressed` | `isotropic_scale_v0` | source name |

A missing `_fidelity` key is treated as pre-Decision-9. The validator
then warns: diastolic filling is unphysiological; do not test E-wave /
IVPG / vortex-ring hypotheses.

**Tell PINNecho consumers now:** beat data from before Decision 9 is
implemented is usable for **systole and ejection only**. The
diastolic filling segment is not physiological.

## Attempt 2 (allowed; will not fix fill)

Retune 2 may run. V0 is unchanged, so filling and EF will not be
solved. Prediction: `Vmax` about **108–110 mL**, order only.

Purpose of this run is limited to:

- valve-event order
- ESV at end-ejection
- AV close

ESV absolute value depends on `T_max`, afterload, and IVC leak
(control). Attach that caveat to any ESV number. This is the second
retune; the third is after Decision 9.

Pre-run check (solver, not this repo): first PV line must show
`R_la 0.03`, `q_close 10`, `end = 0.33`. Keep the laptop awake for
the ~4.8 h run.

## `q_close`

While a valve is open, `p_lv = p_la`, so a pressure-based open/close
switch cannot fire. Close is therefore keyed on volumetric flow `Q`
(`q_close`). That is a plumbing workaround, not a physiological
pressure criterion. It belongs in the export notes (`Q_CLOSE_RATIONALE`
in `ibfe_fidelity.py`).

## Next process (approximate V0)

PINNecho now scales a **reference** mesh and writes the exposed
assumptions. The laptop solver then fills at 5 mmHg and 10 mmHg.

See [`DECISION_9_APPROX.md`](DECISION_9_APPROX.md).

```bash
# monk: python3, from ~/PINNecho01, venv on
python3 scripts/decision9_isotropic_v0.py print-scale
python3 scripts/decision9_isotropic_v0.py scale \
    --vertices /absolute/path/to/real_endo_vertices.csv \
    --out data/decision9_v0_approx
```

## What this repo does not do

- Backward-displacement static solves (formal path; later).
- The IBFE fill itself (atrial pressure at 5 and 10 mmHg — laptop).
- Attempt 2 itself (valve-order run on the laptop).
- Treating (b) or (c) as acceptable for diastolic hypotheses.
- Scaling a simulated `IBFEFrames` export and calling that V0.
