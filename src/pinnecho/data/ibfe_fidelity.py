"""Decision 9 unstressed-reference lock and IBFE ``/meta/fidelity``.

PINNecho's core interest is diastolic filling (E-wave, IVPG, vortex
ring). Those phenomena exist only if the ventricle actually fills by
tens of millilitres. Treating the image mesh as the unstressed
reference (V0 = image volume) produces almost no fill, so those
hypotheses cannot be tested on that beat.

Decision 9 is **locked now** as option (a), not deferred until a third
retune:

* The image is already loaded. Build an unstressed reference.
* **Approximate first:** isotropic scale of the image mesh to V0, then
  fill with atrial pressure toward the image volume. Documented as an
  approximation (wall thins; strain state differs).
* **Formal later (not implemented here):** backward-displacement inverse
  for the unstressed shape that deforms to the image under ``p_img``.
  Each IBFE iteration is a static solve.

``p_img`` is unknown. Mid–late diastole is assumed in 5–10 mmHg; check
both. Klotz V0 ~58 mL is a single-beat empiric with large error;
pat12 is congenital and compliance may differ. The linear scale
``(V0 / V_img)^{1/3}`` is an **exposed assumed parameter** because V0
sets the fill volume.

On-disk this object is ``/meta/fidelity``: NPZ key ``_fidelity``
(JSON) or the manifest ``fidelity:`` block. A missing field is treated
as a pre-Decision-9 image-IBFE beat (systole / ejection only).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple, Union


# Exposed assumed parameters (not measurements).
KLOTZ_V0_ML = 58.0
IMAGE_VOLUME_ML_PAT12 = 105.0
P_IMG_MMHG_ASSUMED: Tuple[float, float] = (5.0, 10.0)
WALL_THICKNESS_IMAGE_MM = 7.4
# 7.4 * (58/105)^{1/3} ≈ 6.1 mm; documented as ~6 mm.

SOURCE_VALUES = frozenset({"manufactured", "image_ibfe", "public_volume"})
DECISION_9_VALUES = frozenset({
    "not_applied", "approx", "formal", "not_applicable",
})
REFERENCE_VALUES = frozenset({
    "image_as_unstressed",
    "isotropic_scale_v0",
    "backward_displacement",
    "manufactured",
    "public_volume",
})

Q_CLOSE_RATIONALE = (
    "Valve close uses volumetric flow Q (q_close), not a pressure "
    "threshold: while the valve is open, p_lv = p_la, so a pressure-based "
    "switch cannot fire."
)

ATTEMPT_2_SCOPE = (
    "Retune 2 (R_la=0.03, q_close=10, end=0.33) does not change V0 and "
    "will not restore filling or EF. Predicted Vmax is about 108-110 mL "
    "(order only). Scope is valve-event order, ESV at end-ejection, and "
    "AV close. ESV absolute value depends on T_max, afterload, and IVC leak."
)

PRE_DECISION_9_NOTE = (
    "Pre-Decision-9 image-IBFE beat: the image mesh is used as the "
    "unstressed reference (V0 equals image volume, about 105 mL). "
    "Filling is almost absent, so there is no physiological E-wave, "
    "IVPG, or vortex ring. Use systole and ejection only. Decision 9 "
    "is locked as (a): treat the image as already loaded and build an "
    "unstressed V0 (approximate isotropic scale first; formal backward "
    "displacement later). Klotz V0 about 58 mL is a single-beat empiric "
    "with large error; pat12 is congenital and compliance may differ. "
    "p_img is unknown; assume mid-late diastole 5-10 mmHg and check both. "
    + Q_CLOSE_RATIONALE + " " + ATTEMPT_2_SCOPE
)

APPROX_NOTE = (
    "Decision 9 approximate unstressed reference: the image mesh is "
    "isotropically scaled to V0, then filled by atrial pressure toward "
    "the image volume. This is an approximation — wall thickness scales "
    "with the mesh (about 7.4 mm to about 6 mm) and the strain state is "
    "not the in-vivo loaded state. Formal backward displacement is not "
    "applied. Klotz V0 is a single-beat empiric with large error; pat12 "
    "is congenital. p_img is an assumed mid-late-diastolic range, not a "
    "measurement. " + Q_CLOSE_RATIONALE
)

FORMAL_NOTE = (
    "Decision 9 formal tag only: backward displacement under p_img is "
    "the intended unstressed-shape inverse. PINNecho does not run that "
    "inverse. Each IBFE iteration is a static solve."
)

DIASTOLE_UNUSABLE_WARNING = (
    "fidelity: diastolic filling is unphysiological on this beat. "
    "Usable phases are systole and ejection only. Do not test E-wave, "
    "IVPG, or vortex-ring hypotheses until Decision 9 (isotropic-scale "
    "V0, or later formal backward displacement) is applied."
)


def isotropic_linear_scale(v0_ml: float, image_volume_ml: float) -> float:
    """Linear isotropic scale that maps image volume to V0.

    ``(V0 / V_img)^{1/3}``. V0 is an exposed assumed parameter: it
    directly sets how much the cavity can fill.
    """
    v0 = float(v0_ml)
    v_img = float(image_volume_ml)
    if v0 <= 0.0 or v_img <= 0.0:
        raise ValueError(
            f"V0 and image volume must be positive, got V0={v0_ml} "
            f"image={image_volume_ml}"
        )
    return (v0 / v_img) ** (1.0 / 3.0)


V0_SCALE_ISOTROPIC = isotropic_linear_scale(KLOTZ_V0_ML, IMAGE_VOLUME_ML_PAT12)
WALL_THICKNESS_APPROX_MM = WALL_THICKNESS_IMAGE_MM * V0_SCALE_ISOTROPIC


@dataclass(frozen=True)
class IBFEFidelity:
    """``/meta/fidelity`` for one IBFE (or public-volume) beat."""

    source: str
    decision_9: str
    diastole_usable: bool
    approximation: bool
    usable_phases: Tuple[str, ...]
    reference: str
    v0_scale: Optional[float] = None
    v0_ml: Optional[float] = None
    image_volume_ml: Optional[float] = None
    p_img_mmhg: Optional[Tuple[float, ...]] = None
    notes: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {
            "source": self.source,
            "decision_9": self.decision_9,
            "diastole_usable": bool(self.diastole_usable),
            "approximation": bool(self.approximation),
            "usable_phases": list(self.usable_phases),
            "reference": self.reference,
            "v0_scale": self.v0_scale,
            "v0_ml": self.v0_ml,
            "image_volume_ml": self.image_volume_ml,
            "p_img_mmhg": list(self.p_img_mmhg) if self.p_img_mmhg is not None else None,
            "notes": self.notes,
        }
        for key, value in self.extra.items():
            if key not in out:
                out[key] = value
        return out


def default_pre_decision_9() -> IBFEFidelity:
    """Missing ``/meta/fidelity`` is a pre-Decision-9 image-IBFE beat."""
    return IBFEFidelity(
        source="image_ibfe",
        decision_9="not_applied",
        diastole_usable=False,
        approximation=False,
        usable_phases=("systole", "ejection"),
        reference="image_as_unstressed",
        v0_scale=None,
        v0_ml=IMAGE_VOLUME_ML_PAT12,
        image_volume_ml=IMAGE_VOLUME_ML_PAT12,
        p_img_mmhg=P_IMG_MMHG_ASSUMED,
        notes=PRE_DECISION_9_NOTE,
    )


def manufactured_fidelity() -> IBFEFidelity:
    """NS-exact synthetic field. Decision 9 does not apply."""
    return IBFEFidelity(
        source="manufactured",
        decision_9="not_applicable",
        diastole_usable=True,
        approximation=False,
        usable_phases=("all",),
        reference="manufactured",
        notes="Manufactured NS-exact field. Decision 9 (unstressed V0) does not apply.",
    )


def public_volume_fidelity() -> IBFEFidelity:
    """Measured 4D-flow / phantom. Decision 9 does not apply."""
    return IBFEFidelity(
        source="public_volume",
        decision_9="not_applicable",
        diastole_usable=True,
        approximation=False,
        usable_phases=("all",),
        reference="public_volume",
        notes=(
            "Public 4D-flow / phantom volume (f = 0). Decision 9 "
            "(unstressed V0) is an image-IBFE reference issue and does "
            "not apply."
        ),
    )


def decision_9_approx_fidelity(
    v0_ml: float = KLOTZ_V0_ML,
    image_volume_ml: float = IMAGE_VOLUME_ML_PAT12,
    p_img_mmhg: Sequence[float] = P_IMG_MMHG_ASSUMED,
) -> IBFEFidelity:
    """Tag for an export that used the isotropic-scale V0 approximation."""
    scale = isotropic_linear_scale(v0_ml, image_volume_ml)
    return IBFEFidelity(
        source="image_ibfe",
        decision_9="approx",
        diastole_usable=True,
        approximation=True,
        usable_phases=("systole", "ejection", "diastole"),
        reference="isotropic_scale_v0",
        v0_scale=scale,
        v0_ml=float(v0_ml),
        image_volume_ml=float(image_volume_ml),
        p_img_mmhg=tuple(float(p) for p in p_img_mmhg),
        notes=APPROX_NOTE,
    )


def decision_9_formal_fidelity(
    p_img_mmhg: Sequence[float] = P_IMG_MMHG_ASSUMED,
    image_volume_ml: float = IMAGE_VOLUME_ML_PAT12,
) -> IBFEFidelity:
    """Tag only. PINNecho does not compute backward displacement."""
    return IBFEFidelity(
        source="image_ibfe",
        decision_9="formal",
        diastole_usable=True,
        approximation=False,
        usable_phases=("systole", "ejection", "diastole"),
        reference="backward_displacement",
        v0_scale=None,
        v0_ml=None,
        image_volume_ml=float(image_volume_ml),
        p_img_mmhg=tuple(float(p) for p in p_img_mmhg),
        notes=FORMAL_NOTE,
    )


def fidelity_from_mapping(raw: Optional[Mapping[str, Any]]) -> IBFEFidelity:
    """Parse a mapping; missing or empty → pre-Decision-9 default."""
    if not raw:
        return default_pre_decision_9()
    source = str(raw.get("source", "image_ibfe"))
    decision_9 = str(raw.get("decision_9", "not_applied"))
    phases = raw.get("usable_phases")
    if phases is None:
        if raw.get("diastole_usable"):
            phases = ("all",)
        else:
            phases = ("systole", "ejection")
    p_img = raw.get("p_img_mmhg")
    known = {
        "source", "decision_9", "diastole_usable", "approximation",
        "usable_phases", "reference", "v0_scale", "v0_ml",
        "image_volume_ml", "p_img_mmhg", "notes",
    }
    extra = {k: v for k, v in raw.items() if k not in known}
    return IBFEFidelity(
        source=source,
        decision_9=decision_9,
        diastole_usable=bool(raw.get("diastole_usable", False)),
        approximation=bool(raw.get("approximation", decision_9 == "approx")),
        usable_phases=tuple(str(p) for p in phases),
        reference=str(raw.get("reference", "image_as_unstressed")),
        v0_scale=_opt_float(raw.get("v0_scale")),
        v0_ml=_opt_float(raw.get("v0_ml")),
        image_volume_ml=_opt_float(raw.get("image_volume_ml")),
        p_img_mmhg=tuple(float(p) for p in p_img) if p_img is not None else None,
        notes=str(raw.get("notes", "")),
        extra=extra,
    )


def dumps_fidelity(fid: IBFEFidelity) -> str:
    return json.dumps(fid.to_dict(), sort_keys=True)


def loads_fidelity(raw: Union[None, str, bytes, Mapping[str, Any], Any]) -> IBFEFidelity:
    """Accept JSON text, a mapping, or a numpy 0-d string array."""
    if raw is None:
        return default_pre_decision_9()
    if isinstance(raw, Mapping):
        return fidelity_from_mapping(raw)
    if hasattr(raw, "item") and not isinstance(raw, (bytes, str)):
        try:
            raw = raw.item()
        except (ValueError, AttributeError):
            raw = str(raw)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return default_pre_decision_9()
        return fidelity_from_mapping(json.loads(text))
    return default_pre_decision_9()


def resolve_fidelity(frames) -> IBFEFidelity:
    """Return ``frames.fidelity``, defaulting a missing field to pre-D9."""
    fid = getattr(frames, "fidelity", None)
    if fid is None:
        return default_pre_decision_9()
    if isinstance(fid, IBFEFidelity):
        return fid
    return loads_fidelity(fid)


def diastole_usable(frames) -> bool:
    return bool(resolve_fidelity(frames).diastole_usable)


def _opt_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)
