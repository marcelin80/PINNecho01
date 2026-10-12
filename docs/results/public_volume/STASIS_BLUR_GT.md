# Corrado — blur-GT Dice upper bound (no train)

σ = 1 voxel Gaussian on the **GT** velocity, then the same persist/jet
cuts as A/B. Not a PINN. Not a thrombus. Do not rerun.

Laptop JSON: `stasis_blur_gt.json`.

Pre-registered reading: if 15 cm/s blur Dice ~ 0.6, PINN failure is a
resolution ceiling. If ~ 0.9 and PINN ~ 0.56, PINN is worse than blur.
The measured value sits **between** those poles.

## Numbers (σ = 1 vox)

| cut | GT jet | blur jet | signed err | Dice | P | R | peak ratio |
|---|---|---|---|---|---|---|---|
| 5 cm/s | 79.74 | 70.38 | −11.7 % | 0.917 | 0.978 | 0.863 | 0.905 |
| 10 cm/s | 30.38 | 26.13 | −14.0 % | **0.911** | 0.985 | 0.847 | 0.892 |
| 15 cm/s | 10.80 | 6.88 | −36.3 % | **0.760** | 0.977 | 0.622 | 0.883 |

## Versus PINN

| cut | blur Dice | A Dice mean | B Dice mean |
|---|---|---|---|
| 10 cm/s | 0.911 | 0.807 | 0.829 |
| 15 cm/s | 0.760 | 0.561 | 0.658 |

10 cm/s: 1-voxel blur barely moves the 30 mL jet (Dice 0.91). A/B at
~0.81 / 0.83 are **below** a high smoothness ceiling. That leak is not
a 2 mm resolution story.

15 cm/s: the 10.8 mL core **is** resolution-sensitive. Blur drops Dice
from 1.0 to 0.76, recall to 0.62, volume −36 %. That is not the ~0.6
pole (not “only resolution”). It is also not the ~0.9 pole. PINN A
(0.56) and B (0.66) both sit **below** 0.76. Extra error is
misplacement: blur precision 0.977, B precision 0.74–0.85.

Peak ratio: blur 0.88–0.90, PINN B 0.80–0.88. Most of the amplitude
bias is compatible with 1-voxel smoothness. PINN is a little lower, and
spatially less clean.

B 15 cm/s millilitres (7.4–8.3 mL, −23 to −32 %) exceed blur volume
(6.88 mL, −36 %) but lose Dice. The extra millilitres are not a better
core.

## Verdict

Not a pure resolution ceiling, not “PINN matches a sharp field.” The
thin core has a 0.76 smoothness cap; PINN does not reach it. The 30 mL
10 cm/s jet’s cap is 0.91; PINN does not reach that either. Sampling
did not close the gap. Do not rerun this σ = 1 control.
