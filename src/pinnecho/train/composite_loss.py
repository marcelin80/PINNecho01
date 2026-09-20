"""Composite PINN loss (spec-aligned) with configurable weights + annealing.

    L = w_data     * L_data            (beam-projected Doppler, single component)
      + w_pde      * L_NS_residual     (continuity + momentum)
      + w_scalar   * L_scalar_residual (residence-time transport)
      + w_bc       * L_boundary        (wall + valve Dirichlet)
      + w_ic       * L_initial         (initial condition)
      + w_periodic * L_cycle_periodicity

All weights are configurable and an optional annealing schedule ramps the PDE
weight up over the first ``pde_warmup_frac`` of training so the network fits data
and boundary conditions before the physics constraint is fully applied (this
avoids the trivial ``u=0`` collapse seen with cold-start physics weighting).

The loss operates on a ``PINNNet`` (``.fields()`` / ``.velocity()``) and a set of
mini-batches supplied as tensors. Terms whose data is unavailable (e.g. a batch
is ``None``) are skipped and contribute zero, so the same object serves the toy
2D case and the eventual full IBFE problem.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from ..physics.ns_residual import navier_stokes_residual_nd
from ..physics.scalar_transport import scalar_transport_residual
from ..bc import boundary_conditions as bc


@dataclass
class LossWeights:
    """Static (base) weights for each loss term."""

    data: float = 1.0
    pde: float = 1.0
    scalar: float = 0.1
    bc: float = 1.0
    ic: float = 0.1
    periodic: float = 0.1


@dataclass
class AnnealSchedule:
    """PDE-weight annealing: linear ramp of ``w_pde`` over the warmup window.

    ``effective_pde(step)`` grows from ``pde_start_frac * w_pde`` at step 0 to the
    full ``w_pde`` at ``pde_warmup_frac * total_steps``.
    """

    enabled: bool = True
    pde_warmup_frac: float = 0.3
    pde_start_frac: float = 0.0
    total_steps: int = 10000

    def pde_scale(self, step: int) -> float:
        if not self.enabled:
            return 1.0
        warm = max(1, int(self.pde_warmup_frac * self.total_steps))
        frac = min(1.0, step / warm)
        return self.pde_start_frac + (1.0 - self.pde_start_frac) * frac


def data_loss_beam(
    velocity_pred: Sequence[torch.Tensor],
    beam_dir: torch.Tensor,
    v_beam_measured: torch.Tensor,
    relative: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Doppler data loss on the BEAM component only (never all components).

    ``velocity_pred`` : sequence of ``dim`` component tensors ``(N, 1)``.
    ``beam_dir``      : ``(N, dim)`` unit beam directions.
    ``v_beam_measured``: ``(N, 1)`` measured beam-projected speed.

    Projects the predicted velocity onto the beam and penalises only that
    scalar -- the defining property of a Doppler PINN.
    """
    preds = torch.cat(list(velocity_pred), dim=1)
    v_beam_pred = (preds * beam_dir).sum(dim=1, keepdim=True)
    res = v_beam_pred - v_beam_measured
    if relative:
        return res.pow(2).mean() / (v_beam_measured.pow(2).mean() + eps)
    return res.pow(2).mean()


@dataclass
class CompositeLoss:
    """Assembles the six-term PINN loss for a ``PINNNet`` model."""

    rho: float
    mu: float
    weights: LossWeights = field(default_factory=LossWeights)
    anneal: AnnealSchedule = field(default_factory=AnnealSchedule)
    diffusivity: float = 1e-6
    scalar_source: float = 1.0
    forcing: Optional[str] = None  # None (Model A) or "fsi" (Model B, future)
    # Residual non-dimensionalisation: multiply raw residuals by these before
    # squaring so continuity / momentum / scalar terms are O(1) and comparable
    # (continuity_scale ~ L/U, momentum_scale ~ L/(rho U^2)). Defaults keep the
    # residuals dimensional (=1.0).
    continuity_scale: float = 1.0
    momentum_scale: float = 1.0
    scalar_scale: float = 1.0

    def __call__(
        self,
        model,
        batches: Dict[str, dict],
        step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss terms.

        ``batches`` maps term name -> a dict of the tensors that term needs:

        * ``"collocation"``: ``{"X": (N, d+1), "forcing": (N, d) or None}``
        * ``"data"``       : ``{"X": (M, d+1), "beam_dir": (M, d), "v_beam": (M,1)}``
        * ``"wall"``       : ``{"X": (Nw, d+1), "u_wall": (Nw, d)}``
        * ``"valve"``      : ``{"X": (Nv, d+1), "u_valve": (Nv, d)}``
        * ``"scalar_inflow"``: ``{"X": (Ni, d+1)}``
        * ``"ic"``         : ``{"X": (Ni, d+1), "velocity": (Ni, d)}``
        * ``"periodic"``   : ``{"X0": (Np, d+1), "XT": (Np, d+1)}``

        Any missing/``None`` batch contributes zero. Returns a dict with the
        weighted ``"total"`` plus each unweighted component for logging.
        """
        w = self.weights
        out: Dict[str, torch.Tensor] = {}
        device_ref = next(model.parameters())
        zero = torch.zeros((), dtype=device_ref.dtype, device=device_ref.device)

        # --- Data (beam-only) ---
        b = batches.get("data")
        if b is not None:
            X = b["X"].requires_grad_(True)
            vel = model.velocity(X)
            out["data"] = data_loss_beam(vel, b["beam_dir"], b["v_beam"])
        else:
            out["data"] = zero

        # --- PDE residual (continuity + momentum) + scalar transport ---
        b = batches.get("collocation")
        if b is not None:
            X = b["X"].requires_grad_(True)
            f = model.fields(X)
            vel = [f[k] for k in (["u", "v", "w"][: model.spatial_dim])]
            forcing = b.get("forcing") if self.forcing == "fsi" else None
            cont, mom = navier_stokes_residual_nd(
                vel, f["p"], X, self.rho, self.mu, forcing=forcing
            )
            pde = (cont * self.continuity_scale).pow(2).mean() + sum(
                (m * self.momentum_scale).pow(2).mean() for m in mom
            )
            out["pde"] = pde
            if model.predict_scalar:
                sc = scalar_transport_residual(
                    f["c"], vel, X, self.diffusivity, self.scalar_source
                )
                out["scalar"] = (sc * self.scalar_scale).pow(2).mean()
            else:
                out["scalar"] = zero
        else:
            out["pde"] = zero
            out["scalar"] = zero

        # --- Boundary (wall + valve) ---
        bc_loss = zero
        b = batches.get("wall")
        if b is not None:
            vel = model.velocity(b["X"])
            bc_loss = bc_loss + bc.wall_kinematic_loss(vel, b["u_wall"])
        b = batches.get("valve")
        if b is not None:
            vel = model.velocity(b["X"])
            bc_loss = bc_loss + bc.valve_dirichlet_loss(vel, b["u_valve"])
        b = batches.get("scalar_inflow")
        if b is not None and model.predict_scalar:
            c = model.fields(b["X"])["c"]
            bc_loss = bc_loss + bc.scalar_inflow_loss(c)
        out["bc"] = bc_loss

        # --- Initial condition ---
        b = batches.get("ic")
        if b is not None:
            vel = model.velocity(b["X"])
            preds = torch.cat(vel, dim=1)
            target = b["velocity"]
            out["ic"] = (preds - target).pow(2).mean() / (target.pow(2).mean() + 1e-12)
        else:
            out["ic"] = zero

        # --- Cycle periodicity ---
        b = batches.get("periodic")
        if b is not None:
            f0 = model.forward(b["X0"])
            fT = model.forward(b["XT"])
            out["periodic"] = (f0 - fT).pow(2).mean()
        else:
            out["periodic"] = zero

        pde_scale = self.anneal.pde_scale(step)
        total = (
            w.data * out["data"]
            + w.pde * pde_scale * out["pde"]
            + w.scalar * pde_scale * out["scalar"]
            + w.bc * out["bc"]
            + w.ic * out["ic"]
            + w.periodic * out["periodic"]
        )
        out["total"] = total
        out["pde_scale"] = torch.as_tensor(pde_scale)
        return out
