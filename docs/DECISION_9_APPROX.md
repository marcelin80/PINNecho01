# Decision 9 next step — isotropic V0, then fill

The lock is already in [`DECISION_9.md`](DECISION_9.md): option **(a)**,
approximate first. This is the process to run now.

PINNecho does the `/meta/fidelity` tag. On monk the **V0 mesh already
exists** — do not scale `LV.e` again. The laptop IBFE solver is in the
fill. Formal backward displacement is still later.

**Applied (2026-10-13, solver homothety, `~/pat12_meshes/scale_reference.json`):**

| quantity | value |
|---|---|
| image | `~/pat12_meshes/LV.e` |
| V0 mesh | `~/pat12_meshes/LV_V0.e` |
| `V_img` | **105.466 mL** (measured cavity) |
| `V0` | **58.000 mL** |
| `alpha` | **0.819292** (`(58/105.466)^{1/3}`) |
| wall | 7.4 mm → 6.06 mm |
| invert | 0; `pass_v0` true |
| this fill | `~/lv_runs/scale_fill`, `P_END=8` mmHg, ramp 0.08 s + hold 0.45 s |

`print-scale` with 58/105 gives 0.8205. That is the rounded formula.
The mesh used the measured 105.466 mL. Keep **0.819292**. 5 and 10 mmHg
are recorded as Klotz-V0 sensitivity (60.12 / 56.95 mL). This first
fill is the midpoint 8 mmHg. Do not abort it to start 5 and 10.

On monk: use **`python3`**, not `python` (that name is not installed).
Run from the repo root (`~/PINNecho01`), not from `~`. Activate
`.venv` so `pinnecho` imports. `path/to/endo_*.csv` is a placeholder —
`print-scale` needs no mesh; `scale` needs the real reference vertices.

```
image mesh (loaded, ~105 mL)
        │  isotropic scale about centroid
        │  s = (V0 / V_img)^{1/3}   ← exposed assumption
        ▼
unstressed V0 (~58 mL, approximation)
        │  atrial-pressure fill
        │  p_img = 5 mmHg  and  10 mmHg
        ▼
image-volume neighbourhood (~105 mL)
        │  export + tag decision_9=approx
        ▼
PINNecho may use diastole (as an approximation)
```

## 0. Attempt 2 may still run

It does not change V0. Filling and EF will not be solved. Use it only
for valve-event order, ESV at end-ejection, and AV close. First PV
line: `R_la 0.03`, `q_close 10`, `end = 0.33`. Keep the laptop awake.

A pre-D9 beat from that run is **systole / ejection only**.

## 1. Print the exposed scale (no mesh required)

On monk:

```bash
cd ~/PINNecho01
git fetch origin cursor/d9-isotropic-v0-935e
git checkout cursor/d9-isotropic-v0-935e
source .venv/bin/activate
python3 scripts/decision9_isotropic_v0.py print-scale \
    --v0-ml 58 --image-volume-ml 105 --p-img-mmhg 5 10
```

Expect `v0_scale` ≈ **0.8205**. If `python` is not found, do not install
`python-is-python3` for this; call `python3`.

Default is Klotz 58 mL / pat12 image 105 mL →
`(58/105)^{1/3} ≈ 0.82`. That number is an assumption. Klotz is a
single-beat empiric with large error; pat12 is congenital. Do not hide
the ratio inside a mesh script.

## 2. Scale the *reference* mesh

Vertices in SI metres (or pass `--millimetres`). Optional faces let the
script measure cavity volume before and after.

```bash
# still in ~/PINNecho01 with .venv on
python3 scripts/decision9_isotropic_v0.py scale \
    --vertices /absolute/path/to/real_endo_vertices.csv \
    --faces /absolute/path/to/real_endo_faces.csv \
    --v0-ml 58 --image-volume-ml 105 \
    --p-img-mmhg 5 10 \
    --out data/decision9_v0_approx
```

Do not run this until the CSV (or `.npy`) actually exists. Faces are
optional; without them the script still scales but cannot measure mL.

Writes `vertices_scaled.csv`, `fidelity.json`, `report.json`,
`report.md`. Centre defaults to the vertex mean.

**Do not** pass a simulated `IBFEFrames` `.npz`. Those points are
already on the loaded trajectory. The script refuses that file.

This is an approximation: wall thickness scales with the mesh
(~7.4 mm → ~6 mm) and the strain state is not the in-vivo loaded
state. `fidelity.json` says so (`approximation: true`).

## 2b. On monk now — let `scale_fill` finish

Do not start PINNecho `scale`. Do not start a second fill at 5 or 10
while this one is running. Power / performance / no-sleep stay on.

```bash
pgrep -af main3d_lv_passive || echo 'solver not running'
tail -3 ~/lv_runs/scale_fill.out
tail -5 ~/lv_runs/scale_fill/lv_pv.dat
```

Watch:

- `V` should leave 58 mL as P rises toward 8 mmHg. Target neighbourhood
  is **~105 mL**, not V0.
- `wall volume ratio` stays ≈ 1.
- `max |J-1|` stays under the 0.8 stop, and does not run away. A few
  elements above 0.1 at 2 mmHg (elem 48946) is a watch item, not a
  kill.
- `box clearance` stays above 0.8 cm.

At step 800 this run was t=0.026 s, P≈2 mmHg, V=58.23 mL — still the
ramp. END_TIME=0.53 s ≈ 16k steps. Do not read Vmax from the first
minutes.

After it exits: last PV line is the 8 mmHg hold. If V is far below
105 mL, the lever is V0 or the HO law, not `q_close`. Then consider
the 5 and 10 mmHg fills. If `max |J-1|` climbed through the stop,
do not export.

## 3. Solver: fill at 5 mmHg and 10 mmHg

`p_img` is unknown. Phase is not locked. Mid–late diastole is assumed
in 5–10 mmHg.

1. Replace the IBFE unstressed reference with `vertices_scaled.*`.
2. Fill with atrial pressure toward the image volume.
3. Run **two** fills (or two short beats): `p_la` / image pressure
   **5 mmHg** and **10 mmHg**.
4. Record V at each. The target neighbourhood is the image volume
   (~105 mL), not V0. If both miss by a large amount, the V0
   assumption is the lever — not another valve retune.
5. Keep laptop power / performance / no-sleep for the long run.

Do not start a third retune of `T_max` / `q_close` / `R_la` until this
fill exists. Attempt 2 is the last pre-D9 retune.

## 4. Export and tag

Copy `fidelity.json` into the export (`_fidelity` NPZ key or manifest
`fidelity:`), or tag after the fact:

```bash
python3 scripts/decision9_isotropic_v0.py tag \
    --export /absolute/path/to/new_beat.npz \
    --v0-ml 58 --image-volume-ml 105 \
    --p-img-mmhg 5 \
    --out /absolute/path/to/new_beat_d9.npz
```

Use the `p_img` of **that** run (5 or 10), not both, on the tagged
file. `diastole_usable` becomes true, still marked `approximation`.

`validate_ibfe_frames` will stop warning that filling is
unphysiological. E-wave / IVPG / vortex-ring work may start on these
beats only, and must still say “Decision 9 approx”.

## What not to do

- Do not scale a solved beat and call that V0.
- Do not implement backward displacement in this repository.
- Do not treat a pre-D9 export as diastolic GT.
- Do not quote persist Dice or start a thrombus solver.
