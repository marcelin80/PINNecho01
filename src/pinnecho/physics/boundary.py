"""Wall boundary-condition residuals for the moving endocardium.

Both physics backbones enforce a no-slip condition on the moving wall: the
fluid velocity must equal the prescribed wall (endocardial) velocity. The
*source* of that prescribed velocity is what differs conceptually:

* baseline      -- wall velocity taken directly from the segmented / kinematic
  endocardial contour (as in AI-VFM, iVFM-PINN, CSF-PINN);
* fsi_informed  -- wall velocity taken from the FSI simulation.

In the Stage-1 synthetic setting the manufactured solution satisfies exact
no-slip, so the two wall velocities coincide and the backbones are separated
purely by the FSI body-forcing term in the momentum equation. Keeping this a
dedicated module means a future stage can diverge the two wall velocities
(e.g. adding structural normal traction) without touching the trainer.
"""

import torch


def wall_bc_residual(
    u_pred: torch.Tensor,
    v_pred: torch.Tensor,
    wall_velocity: torch.Tensor,
) -> torch.Tensor:
    """No-slip residual ``[u - u_wall, v - v_wall]`` stacked as ``(N, 2)``."""
    res_u = u_pred - wall_velocity[:, 0:1]
    res_v = v_pred - wall_velocity[:, 1:2]
    return torch.cat([res_u, res_v], dim=1)
