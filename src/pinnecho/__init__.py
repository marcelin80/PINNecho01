"""PINNecho: FSI-informed PINN for intraventricular Doppler flow reconstruction.

Stage 1 operates entirely on *synthetic* data that mimics the output of an
IBAMR/IBFE cardiac fluid-structure-interaction (FSI) pipeline. The package lets
you train and compare two physics backbones for reconstructing full-field
velocity, pressure, vorticity and blood residence time from sparse,
single-component (Doppler-like) velocity samples:

* ``baseline``  -- generic incompressible Navier-Stokes with a purely
  *kinematic* no-slip wall boundary condition (wall velocity prescribed from the
  moving endocardial contour). This mirrors AI-VFM / iVFM-PINN / CSF-PINN.
* ``fsi_informed`` -- the same Navier-Stokes residual, but additionally informed
  by the FSI simulation's body-forcing term (net effect of myocardial active
  contraction and structural coupling) that the baseline ignores.

See the top-level ``README.md`` for the scientific motivation.
"""

from .config import Config, load_config

__all__ = ["Config", "load_config"]
__version__ = "0.1.0"
