# Do not use this IBFE series for E-wave, IVPG, or physiology

**Use: `pipeline-validation` only.**

This export may be used to check the IBFE → PINNecho pipe (valve-event
order, that filling exists, that the on-disk contract loads). It must
**not** be used to evaluate an E-wave, an IVPG, a vortex ring, or any
physiological interpretation of diastole. E-wave / IVPG remain the
PINNecho goal. This series cannot support that claim.

`/meta/fidelity.purpose = pipeline-validation`.
`diastole_usable = false`. `e_wave_ivpg_usable = false`.
`filling_present` may be true. Do not retag a released file.

---

## Close-out decision: no Eulerian dump, no oracle A/B

The present files are wall data (`eulerian_present = 0`). Oracle A/B
needs `velocity_fluid` (and, for Model B, band-localized `forcing_fluid`).

PINNecho has **no** converter from IBAMR viz / restart / Silo into an
`IBFEFrames` Eulerian field. Building that path from beat-2 output is
not a same-day job in this repository. Expected gain is low: real IB
`f` is the structural force spread in a thin wall band (cavity interior
exactly 0), and Ablation 7 already showed that putting true band `f`
into the oracle **worsened pressure** (relL2 **13.28 ± 3.39** vs
baseline **1.06 ± 0.02**, 0/3 seeds). See [`ABLATIONS.md`](ABLATIONS.md)
Ablation 7.

**Locked:** do not convert the velocity field; do not run oracle A/B
on true IBFE `f`. Record: *the oracle test on true IBFE `f` was not
done.*

---

## Model state (as reported; not a PINNecho measurement)

- Uniform active stress (not a spatially varying activation map).
- One 16-year-old congenital **mild** case (pat12). Not a failing /
  dilated cohort.
- EF **about 40%**. This is the closer's model-state figure. It is
  **not** locked to a corrected ED3. The warmup beat had EF 33% at
  V0 = 58 mL (ED 87.7 mL). Do not treat either number as the ED3 EF
  until the MV-close script reports.
- Ejection duration **106 ms — literature check required**, not
  confirmed here.
- Diastolic **chatter suspected**, not confirmed. Confirm with the
  MV-close script before writing it as a finding.

## Not yet determined — do not write as fact

| item | status |
|---|---|
| Corrected ED3 | waiting on MV-close script |
| Chatter present / absent | waiting on MV-close script |
| Adoptable `/events` time window (seconds) | waiting on export README values |
| Ejection 106 ms vs literature | not checked here |

When those values exist, paste them into this table. Do not retag the
NPZ to carry them.

---

## Known export error: `/events`

`/events` currently points at **beat 1's first transition**. That will
not be fixed by retagging.

**Consumer warning (put on the export README, with numbers):**

- Do not use `/events` as the beat-2 / beat-3 window.
- File re-tag is forbidden. Do not rewrite `_fidelity` or `/events` on
  a released bundle to “fix” the pointer.
- Adoptable time window, seconds, copied from the solver README once
  the MV-close script reports (do not invent):

```
t_adopt_start_s:  <not filled>
t_adopt_end_s:    <not filled>
```

PINNecho loaders do not interpret `/events`. Downstream scripts that
do must read the README window, not `/events`.

---

## Not provided in this export

- Eulerian velocity (`velocity_fluid` / `eulerian_present = 0`)
- `traction_wall`
- Per-beat valve-transition counts

Wall kinematics may be present. That is not an oracle fluid field.
A wall-only bundle fails `validate_ibfe_frames` (`coords_fluid`
empty). That is the training contract. Valve-order plumbing is
solver-side, not a PINNecho train path.

---

## Estimates — not measurements

Mark these as **estimates** wherever they are quoted:

- ~12 h/s *(estimate)*
- ~190 h/s *(estimate)*
- Band-coverage fraction at `dx = 1.875 mm` of **43–59%** of cavity
  volume *(estimate from the Minimum Goal B resolution note; depends
  on geometry and kernel width)*

Do not promote them to timings or to a measured band fraction for
this mesh.

---

## Not tested

- Oracle A/B on true IBFE `f` *(skipped; see close-out decision)*
- E-wave / IVPG / vortex-ring evaluation *(not admissible on this
  series; needs gate R `dx = 0.94 mm` and a fill that can be used as
  physiology)*
- Grid convergence at `dx = 0.94 mm` (gate R)

Minimum Goal B (`dx = 1.875 mm`) is plumbing. It is not a scientific
resolution for pressure gradients.

---

## Reproducibility

**PINNecho (this document):**

- Branch: `cursor/ibfe-close-limitations-935e`
- Commit: filled after the introducing commit lands (do not invent)
- Environment: Python ≥ 3.10, CPU PyTorch, `pip install -e ".[dev]"`
- Command: `pytest tests/test_ibfe_fidelity.py -q`
- Code: `pipeline_validation_fidelity()` in
  `src/pinnecho/data/ibfe_fidelity.py`
- Tests: `tests/test_ibfe_fidelity.py`

**IBFE export (copy from the solver README; required; do not invent):**

- Solver commit / IBAMR / PETSc / SAMRAI / MPI versions
- Launch command and `input3d`
- Mesh: `~/pat12_meshes/LV_V0.e` (Decision 9 a1, α = 0.819292,
  V0 = 58.000 mL, V_img = 105.466 mL)
- SHA-256 of the released wall bundle and of `scale_reference.json`

Until those hashes are pasted here, this repository records the
contract and the limits, not the binary provenance.

---

## IBAMR issue (solver-side, not this repo)

PINNecho does not file the IBAMR abort-at-exit issue and did not
apply a patch or rebuild IBAMR. That check is on monk. Before anyone
files it:

1. Apply the patch and rebuild **only that source**; confirm the
   abort at exit is gone.
2. Check whether the latest release still reproduces it, and whether
   an existing issue already covers it.
3. The report must include: ex4 minimal reproduction (name IB vs
   IBFE from the abort stack); versions (IBAMR, PETSc, SAMRAI, MPI);
   the patch diff; and whether the patch is a **root fix** or a
   **symptom workaround**.

**Duplicate / latest-release scan (this repo, 2026-10-28, America/New_York):**

- Latest IBAMR release: **v0.19.0** (2026-03-11 UTC). Reproduction
  on that tag is **not done here**.
- No open `IBAMR/IBAMR` issue matched abort-at-exit /
  `PetscFinalize` destructor order. Closed [#290](https://github.com/IBAMR/IBAMR/issues/290)
  is missing input keys in tests, not exit abort. Closed
  [#1771](https://github.com/IBAMR/IBAMR/issues/1771) is a
  libMesh/PETSc **build**, not exit abort.
- Do not file until step 1 is done on monk and v0.19.0 (or
  `master`) is checked.

---

## Paper strength (PINNecho evaluation draft)

Recorded so it is not “someone else's job” and then omitted. Based on
summaries, not a new read of the raw runs.

- Observability is not new (triplane iVFM).
- Persist Dice ~0.96 is high because persist is ~84% of the cavity.
  Leading with that number as evidence of a metric failure can be
  dismissed as obvious.
- Jet under-estimation overlaps a known smoothing prior (PINN
  spectral bias). Literature, checked 2026-10-28:
  Rahaman et al., ICML 2019 (*On the Spectral Bias of Neural
  Networks*, PMLR 97:5301–5310);
  Wang, Yu and Perdikaris, *J. Comput. Phys.* 449:110768 (2022)
  (*When and why PINNs fail to train: A neural tangent kernel
  perspective*);
  Wang, Wang and Perdikaris, *Comput. Methods Appl. Mech. Eng.*
  384:113938 (2021) (*On the eigenvector bias of Fourier feature
  networks*). Cite them; do not present the leak as a newly
  discovered PINN law. This repo keeps random Fourier features
  (Tancik et al., 2020) **off by default** because they corrupt
  derivatives; that lever was **not** ablated on the Corrado jet.
- Data: one rigid phantom, three seeds, one method.

The defensible contribution is narrow: **quantify the mismatch
between global metrics and a downstream-task metric, and propose an
evaluation protocol.** That needs the same direction on a classical
baseline and on DUPLEX, with ≥ 5 seeds. Journal acceptance after
review is unlikely (probability unknown). The surest product is a
preprint plus public code and evaluation tools.

**Stop rule (fix the cap before seeing millilitres):** 4–6 weeks
upper bound. If the jet loss is **not** reproduced on a classical
baseline or on DUPLEX, cut the manuscript to a technical note. Do
not train DUPLEX (public dump has no `(u,w)`). Do not rerun frozen
Corrado A/B/C.

---

## Pending paste-ins (from the solver)

Send these when they exist; they will be appended as values, not as
a file retag:

1. MV-close script output and corrected **ED3**
2. Chatter present / absent from that script
3. Numeric `t_adopt_start_s` / `t_adopt_end_s`
4. SHA-256 of the released bundle
