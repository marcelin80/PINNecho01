# The wall is the question, not GPU or data

Locked 2026-11-09. Root cause is **not** missing GPU and **not**
missing patient / IBFE data. This project asked a **different
problem, with different numbers**, than the VFM/PINN references, and
that problem is not what PINN + Navier–Stokes is good at.

Frozen 2k A/B/C tables stay in
[`STASIS_RECON_SEEDS.md`](results/public_volume/STASIS_RECON_SEEDS.md).
Do not rerun them. 20k figures below are **reported** (2026-11-09);
JSON is not in this repo. Do not rerun 2k to “match” them.

## What the references actually asked

AI-VFM, iVFM-PINN, CSF-PINN, triplane iVFM are mostly **2-D (or
triplane) colour-Doppler VFM**. The typical exhibit is a visible
streamline / vortex, a velocity RMSE, and sometimes WSS. CSF-PINN
(Wong et al. 2025) already wrote that gradients and WSS are weak.
Those papers do **not** ask to match a **15 cm/s jet volume against
4D-flow GT** (10.8 mL; at 2 mm that is about **1350 voxels**).

## Question order that died

1. Does FSI `f` improve pressure / gradients? **Rejected**
   (circularity: oracle `f` tracks `∇p`). That was the axis that
   was supposed to differ from the references. It died first.
2. Do sparse one-component windows restore a **3-D volume**?
   Extra windows bring back unseen components. That is
   **observability triplane iVFM already did**. Not a title.
3. Does the **15 cm/s core / stasis remainder** match? **This is
   the wall.**

## Today’s numbers split the wall

| piece | 2k Adam (frozen A, 3 seeds) | 20k (reported, 2 seeds) |
|---|---|---|
| speed relL2 | ~0.42 | **~0.08** |
| 10 cm/s jet signed error | −14 / −20 / −16 % | **≈ 0 %** |
| 15 cm/s jet signed error | −26 / −51 / −48 % | **−21 ~ −23 %** |
| 15 cm/s peak ratio | ~0.80–0.88 (B @ 2k) | **0.921 and 0.947** |

- **Resource explains:** 10 cm/s millilitres and relL2. “We trained
  less than the papers” is true **only this far**.
- **Resource does not explain:** the 15 cm/s core. Two seeds at 20k
  still lose 21–23 %. Reported Fourier-1 is **−70 %** on that core.
  Blur-GT 15 cm/s Dice is **0.760** ([`STASIS_BLUR_GT.md`](results/public_volume/STASIS_BLUR_GT.md)):
  **one voxel of smear already breaks this set.** PINN+NS is
  essentially a low-pass; the tail cut goes first.

Do not promote persist Dice ~0.96 or the recovered 10 cm/s row as
evidence that the 15 cm/s question is solved.

## Data is the easier side of this wall

Corrado is a rigid pump, 2 mm, 20 frames, `f = 0`. That is **easier**
than patient echo, not harder. PIV cannot validate: the jet is
outside its FOV. Missing patient volumes / real IBFE means the
**original FSI and thrombus story cannot proceed.** That is not why
the 15 cm/s core failed. It failed **with** a 4D-flow GT in hand.
Harder data will not lower this wall.

## One line

The references wrote papers on **showing large-scale 2-D
streamlines**. This project asked for **3-D millilitres of a small
fast core**. Using less GPU only made 10 cm/s look ugly. The real
wall is the task.
