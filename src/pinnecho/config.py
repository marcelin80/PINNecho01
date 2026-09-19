"""Configuration schema and YAML (de)serialisation for PINNecho.

The configuration is a set of nested :class:`dataclasses` so that experiments
are fully described by a single YAML file. Anything not present in the YAML
falls back to the defaults defined here, which keeps config files small and
readable while remaining explicit about every knob that exists.
"""

import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class GeometryConfig:
    """Moving left-ventricle (LV) cavity geometry.

    The synthetic LV cavity is a time-deforming ellipse. ``r0_x`` / ``r0_y`` are
    the mean semi-axes (metres). ``strain_amplitude`` controls the peak
    fractional shortening of the long axis over the cardiac cycle; the short
    axis compensates so the 2D cavity *area is preserved* (2D incompressibility).
    """

    r0_x: float = 0.020
    r0_y: float = 0.030
    strain_amplitude: float = 0.18
    center_x: float = 0.0
    center_y: float = 0.0


@dataclass
class FlowConfig:
    """Manufactured intracavitary flow field parameters.

    The ground-truth velocity is the sum of an area-preserving wall-following
    (affine) field and a divergence-free interior vortex derived from a stream
    function that vanishes together with its gradient on the wall (exact
    no-slip). ``vortex_strength`` scales the interior swirl, ``inflow_strength``
    the diastolic filling jet, and ``pressure_amplitude`` the manufactured
    pressure field (Pa).
    """

    period: float = 0.9  # cardiac cycle length [s]
    vortex_strength: float = 0.35
    inflow_strength: float = 0.25
    pressure_amplitude: float = 400.0
    density: float = 1060.0  # blood density [kg/m^3]
    viscosity: float = 3.5e-3  # dynamic viscosity [Pa s]


@dataclass
class DopplerConfig:
    """Virtual Doppler acquisition (sparse, single-component samples).

    A virtual transducer sits at ``transducer`` (metres). Each measurement is
    the projection of the true velocity onto the beam direction (transducer ->
    sample point), emulating the radial velocity a color-Doppler / PW-Doppler
    system records. ``n_points_per_frame`` samples are drawn per time frame,
    ``n_frames`` frames span (part of) the cardiac cycle, and Gaussian noise of
    relative std ``noise_level`` is added.
    """

    # Primary virtual transducer (single acoustic window).
    transducer: tuple[float, float] = (0.0, -0.075)
    # Optional additional acoustic windows. Real echocardiography often combines
    # views (e.g. apical + parasternal); with a single window the cross-beam
    # velocity component is poorly observable, so extra windows at different
    # angles make the full 2D velocity recoverable. Each measurement remains
    # single-component (projected onto its own beam). Empty => single window.
    transducers: tuple = ()
    n_points_per_frame: int = 300
    n_frames: int = 12
    t_start_frac: float = 0.05
    t_end_frac: float = 0.55
    noise_level: float = 0.05
    seed: int = 0

    def all_windows(self) -> tuple:
        return tuple(self.transducers) if self.transducers else (self.transducer,)


@dataclass
class CollocationConfig:
    """Point sets used to enforce the PDE and boundary residuals."""

    n_interior: int = 4000
    n_wall: int = 800
    n_frames: int = 16
    resample_every: int = 0  # 0 => fixed point set for the whole run


@dataclass
class ModelConfig:
    """Fourier-feature MLP for the (u, v, p) field."""

    hidden_width: int = 128
    hidden_depth: int = 5
    activation: str = "tanh"
    # Plain tanh MLP by default (fourier_features = 0): smooth derivatives are
    # essential for recovering vorticity / WSS. Random Fourier features are
    # available but disabled by default -- see models/mlp.py and the README.
    fourier_features: int = 0
    fourier_scale: float = 1.0
    output_pressure: bool = True


@dataclass
class PhysicsConfig:
    """Which physics backbone to use.

    ``backbone`` is either ``"baseline"`` (kinematic no-slip only, forcing
    assumed zero) or ``"fsi_informed"`` (adds the FSI body-forcing term to the
    momentum residual). ``forcing_scale`` lets you sweep how strongly the FSI
    forcing is trusted (1.0 = full FSI forcing, 0.0 collapses to baseline).
    """

    backbone: str = "baseline"
    forcing_scale: float = 1.0
    enforce_continuity: bool = True


@dataclass
class LossWeights:
    # All four terms are dimensionless/relative and O(1), so weights are directly
    # comparable. Data + wall are emphasised so both backbones genuinely fit the
    # (sparse, single-component) measurements and the moving-wall no-slip BC; the
    # backbones are then separated by how well the momentum residual is satisfied.
    data: float = 10.0
    continuity: float = 20.0
    momentum: float = 5.0
    wall: float = 10.0


@dataclass
class TrainConfig:
    iterations: int = 4000
    lr: float = 2.0e-3
    lr_decay: float = 0.5
    lr_decay_every: int = 2000
    batch_interior: int = 2048
    batch_wall: int = 512
    log_every: int = 500
    # Fraction of training over which the *physics* (continuity + momentum)
    # weights are ramped linearly from 0 to their full values. At random
    # initialisation both physics residuals are orders of magnitude larger than
    # the (relative) data term AND are both minimised by the trivial u=0 field
    # (which is divergence-free and, with p=const, has zero momentum residual
    # for the baseline). Without a ramp the optimiser collapses to u=0 and never
    # fits the sparse data. Warming up on data + wall first establishes a
    # non-trivial velocity field that fits the Doppler measurements; the physics
    # then refines it (and is where the baseline vs FSI difference emerges).
    physics_warmup_frac: float = 0.3
    # Optional full-batch L-BFGS polishing after Adam. L-BFGS is very effective
    # at driving PINN residuals down the last 1-2 orders of magnitude once Adam
    # has found a good basin; 0 disables it.
    lbfgs_iters: int = 0
    weights: LossWeights = field(default_factory=LossWeights)


@dataclass
class Config:
    experiment: str = "baseline"
    seed: int = 0
    device: str = "cpu"
    # Non-dimensionalisation reference scales (set from geometry/flow at build).
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    doppler: DopplerConfig = field(default_factory=DopplerConfig)
    collocation: CollocationConfig = field(default_factory=CollocationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)


def _build_nested(cls, data: Dict[str, Any]):
    """Recursively construct a dataclass ``cls`` from a plain ``dict``."""
    if not dataclasses.is_dataclass(cls):
        return data
    kwargs: Dict[str, Any] = {}
    fields = {f.name: f for f in dataclasses.fields(cls)}
    for key, value in (data or {}).items():
        if key not in fields:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        f = fields[key]
        f_type = f.type
        if dataclasses.is_dataclass(f_type) and isinstance(value, dict):
            kwargs[key] = _build_nested(f_type, value)
        elif isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    """Load a :class:`Config` from a YAML file, filling in defaults."""
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    return _build_nested(Config, raw)


def config_from_dict(data: Dict[str, Any]) -> Config:
    return _build_nested(Config, data)
