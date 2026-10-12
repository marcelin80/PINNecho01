# Charité inquiry only (do not download, do not run CFD)

Wiegand, Vellguth et al., *Ann Biomed Eng* 2026,
[10.1007/s10439-026-04205-7](https://doi.org/10.1007/s10439-026-04205-7).
Figshare supplement is **private**. Paper license is CC BY-NC-ND; data license
is not stated. Volunteer LV 4D-flow in the paper: 25 frames,
1.96×1.96×2.8 mm — not posted. Their CFD vs MRI aortic-valve ΔP was
7.3 vs 1.6 mmHg; do not take the simulation as GT.

Send this **now**. OSU rest #11 is k-space only; monk has no MATLAB, so
CORe cannot run. Do not start Fluent / STAR-CCM+ / LDDMM.

To: katharina.vellguth@dhzc-charite.de
Cc: moritz.wiegand@dhzc-charite.de
Subject: Data inquiry — LV 4D-flow and mesh correspondence (Wiegand et al., ABME 2026)

```
Dear Dr. Vellguth and Dr. Wiegand,

We are working on reconstructing 3-component LV velocity from sparse
Doppler-like observations (PINN), and we would like to evaluate the
method on a moving-wall LV rather than a rigid phantom.

We read Wiegand et al., Ann Biomed Eng 2026
(doi:10.1007/s10439-026-04205-7). The figshare supplement appears to be
private. Before we request access or attempt any CFD reproduction, could
you please clarify:

1. License — under what terms the mesh / cine / any velocity data can be
   used for a non-commercial research reconstruction paper?
2. Mesh correspondence — are the time-resolved LV surface vertices in
   1-1 correspondence across the cycle (or is a registration required)?
3. 4D-flow velocity — the paper reports a volunteer LV 4D-flow series
   (25 frames, 1.96 × 1.96 × 2.8 mm). Is a de-identified velocity volume
   (plus cavity mask) shareable under a research agreement? This would
   be far more useful to us than re-running CFD.
4. Pressure boundary-condition curves — if a CFD replay is the only
   option, are the inlet/outlet pressure (or Windkessel) time series
   available with the mesh?

We would cite the paper and follow any required data-use agreement.
Thank you for considering this.

Kind regards,
```

If they share the 4D-flow volume, convert with the public-volume contract
(`f = 0`). Do not train `fsi_informed`. If they only share the mesh, wait
until the OSU gate fails before building a solver.
