"""A closed form mechanics model of the same beam, as the independent cross check.

Build spec 13.4 and salvage item 6. The predecessor shipped a script it called a physics
informed analytical propagation model, and this module is the clean reimplementation of that
idea: a model of the beam of build spec 6.2 written from mechanics rather than from data,
pushed through the same input distributions with the same seeded generator, so the surrogate's
propagated peak load has something to be compared against that shares none of its machinery.

**What the salvaged script actually computed, and what is kept.** It evaluated the tip
deflection of the member under a fixed 50 kN load from ``7 P L**3 / (96 E I)`` with
``L = 1.6`` m and ``I = 1.958e-4`` m**4, then divided by a strength factor
``1 - 0.15 (28 - fcm) / 28`` clipped to ``[0.7, 1.1]``. Three of its ingredients are real and
are kept, with the arithmetic redone here:

* the geometry. ``I`` is the gross second moment of the 250 by 150 mm section to three
  figures, and 1.6 m is the pin to load distance of the model, so the script had the member
  in front of it even though the structural system it wrote down is not the one the model has.
* the idea of an independent forward model driven by the same inputs.
* the elastic modulus as the Eurocode 2 function of the strength, which is
  :func:`ufem.config.derived_E` here rather than a second column of a spreadsheet.

**What is not kept, and why.** ``7 P L**3 / (96 E I)`` is the deflection of no standard case:
the propped cantilever with a central load deflects ``7 P L**3 / (768 E I)`` under it, a
factor of eight away, and in any event the model of build spec 6.2 is a simply supported span
with an overhang and is statically determinate, so no propped cantilever formula applies to
it. The strength factor is an invented soft coupling with no mechanical content, and applying
it on top of a modulus that is already a function of the strength counts the same dependence
twice. The fixed 50 kN load makes the output a deflection under an arbitrary load rather than
a capacity, so it cannot be compared with anything the campaign measured. And E and the
strength entered the script as two independent columns, which is the collinearity defect of
build spec 5.4 in its original habitat.

**The structural system.** Build spec 6.2: a 2000 by 250 mm plane stress member of 150 mm
thickness, pinned at (200, 250), on a roller at (1000, 0), with the displacement imposed at
(1800, 250). Vertical load at one point, three reaction components, three equations: the
system is determinate, the horizontal reaction is zero because nothing else carries one, and
the bending moment is largest at the roller with magnitude ``P a`` where ``a`` is the 800 mm
from the roller to the load. The whole member is in hogging, so the tension face is the top
one and the tension reinforcement is the top layer, whose distance from the compression face
at the soffit is exactly the top cover ``c_top``. That is why the analytic capacity depends on
``c_top`` with an elasticity near one: it is the effective depth, not a cover in the usual
sense.

**What the model claims and what it does not.** It is a plane section rectangular stress block
capacity plus a Timoshenko elastic stiffness. It does not model tension softening, damage,
localization, or the mesh, so it is a statement about the section rather than about the finite
element analysis, and :data:`MODEL_ERROR_FRACTION` is the band inside which that statement is
offered. Every function here takes and returns SI units in the project's convention: force in
N, length in mm, stress in MPa, moment in N mm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: The relative model error the comparison of build spec 13.4 is read at, declared before the
#: comparison was run rather than fitted to its outcome.
#:
#: A rectangular stress block against a nonlinear finite element analysis of the same section
#: is conventionally quoted to about 10 percent. Four named effects push it wider here, and
#: they do not cancel: the block ignores the concrete tensile contribution, which is not small
#: in a member whose finite element model defines tension softening without a fracture energy;
#: the finite element peak occurs at 11 mm of imposed displacement, by which point the damaged
#: plasticity compression response has left its plateau; the mesh is 637 reduced integration
#: quadrilaterals over the whole member, which is coarse for a bending gradient; and the
#: viscoplastic regularization at 8e-4 adds a small rate dependent overshoot. Fifteen percent
#: is the band those four justify. It is a stated tolerance on a comparison, never a knob:
#: changing it changes what this project claims, so it changes only with a design decision.
MODEL_ERROR_FRACTION = 0.15


@dataclass(frozen=True)
class Beam:
    """The member of build spec 6.2, as mechanics rather than as an input file.

    Lengths in mm, stresses in MPa. ``span_mm`` is the pin to roller distance and
    ``overhang_mm`` the roller to load distance, so the pin to load distance is their sum.
    Both cover values are measured from the soffit, matching
    ``configs/probabilistic_model.yaml`` and the geometry validator in :mod:`ufem.config`.
    """

    span_mm: float
    overhang_mm: float
    depth_mm: float
    thickness_mm: float
    n_bars_top: int
    bar_diameter_top_mm: float
    n_bars_bottom: int
    bar_diameter_bottom_mm: float
    steel_yield_MPa: float
    steel_modulus_MPa: float
    concrete_ultimate_strain: float
    block_depth_factor: float
    block_stress_factor: float
    poisson_ratio: float
    shear_correction: float

    @property
    def area_top_mm2(self) -> float:
        return self.n_bars_top * 0.25 * np.pi * self.bar_diameter_top_mm**2

    @property
    def area_bottom_mm2(self) -> float:
        return self.n_bars_bottom * 0.25 * np.pi * self.bar_diameter_bottom_mm**2

    @property
    def section_area_mm2(self) -> float:
        return self.thickness_mm * self.depth_mm

    @property
    def gross_second_moment_mm4(self) -> float:
        """``b h**3 / 12`` of the uncracked rectangle, steel not transformed."""
        return self.thickness_mm * self.depth_mm**3 / 12.0

    @property
    def steel_yield_strain(self) -> float:
        return self.steel_yield_MPa / self.steel_modulus_MPa


#: The beam the inherited campaign actually ran, from build spec 6.2. The two block factors
#: are the Eurocode 2 rectangular stress block for concrete classes up to C50/60 (``lambda``
#: 0.8 and ``eta`` 1.0); the ultimate compressive strain is the same code's 0.0035; the
#: Poisson ratio and the shear correction factor are the standard concrete value and the
#: rectangular section value. None of these is a distribution parameter: they are constants of
#: the mechanics, and the three random inputs reach every function here as arguments.
DEFAULT_BEAM = Beam(
    span_mm=800.0,
    overhang_mm=800.0,
    depth_mm=250.0,
    thickness_mm=150.0,
    n_bars_top=3,
    bar_diameter_top_mm=12.0,
    n_bars_bottom=3,
    bar_diameter_bottom_mm=10.0,
    steel_yield_MPa=500.0,
    steel_modulus_MPa=200000.0,
    concrete_ultimate_strain=0.0035,
    block_depth_factor=0.8,
    block_stress_factor=1.0,
    poisson_ratio=0.2,
    shear_correction=5.0 / 6.0,
)

#: Relative tolerance the equilibrium residual of the solved neutral axis must clear.
EQUILIBRIUM_TOLERANCE = 1.0e-9


class SectionAssumptionViolated(ValueError):
    """The section is outside the regime the rectangular block formula describes."""


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def compression_steel_stress_MPa(
    neutral_axis_mm: Any, bar_depth_mm: Any, beam: Beam = DEFAULT_BEAM
) -> np.ndarray:
    """Stress in a bar layer at ``bar_depth_mm`` from the compression face, positive in
    compression.

    Plane sections: the strain varies linearly from ``concrete_ultimate_strain`` at the
    compression face to zero at the neutral axis, so a layer above the neutral axis is in
    compression and one below it is in tension, and the sign follows from the geometry rather
    than from an assumption about which happened. The result is clipped at the yield stress in
    both directions, which is the elastic perfectly plastic steel of build spec 6.2.
    """
    x = _as_array(neutral_axis_mm)
    depth = _as_array(bar_depth_mm)
    if np.any(x <= 0.0):
        raise SectionAssumptionViolated(
            f"the neutral axis depth must be positive, got a minimum of {float(x.min())}."
        )
    strain = beam.concrete_ultimate_strain * (x - depth) / x
    return np.clip(beam.steel_modulus_MPa * strain, -beam.steel_yield_MPa, beam.steel_yield_MPa)


def neutral_axis_depth_mm(
    fcm_MPa: Any, c_bottom_mm: Any, beam: Beam = DEFAULT_BEAM
) -> np.ndarray:
    """Depth of the neutral axis from the soffit, in mm, solved in closed form.

    The top cover is not an argument on purpose. Once the tension steel is taken as yielded,
    its force is ``As fy`` wherever it sits, so horizontal equilibrium does not see its depth;
    the depth enters the lever arm in :func:`hogging_capacity` and nowhere else. Passing it
    here would suggest an influence the equation does not have.

    Horizontal equilibrium of the hogging section is

        ``lambda x b eta fcm + As_bot sigma_s(x) = As_top fy``

    with the tension steel taken as yielded, which :func:`hogging_capacity` then verifies. The
    bottom layer sits close to the compression face, so its stress is the part that decides
    which of three regimes the section is in, and each regime has an explicit root:

    * bottom layer elastic. Multiplying through by ``x`` gives a quadratic whose positive root
      is the answer, and it is unique because the constant term is negative.
    * bottom layer yielded in compression, near the top of the range: the equation is linear.
    * bottom layer yielded in tension, which happens when the neutral axis rises above it:
      linear again, with the layer's force on the other side.

    Solving all three and selecting by the stress the elastic root implies is exact and
    vectorizes, which matters because the propagation evaluates this at 1e5 samples. The
    equilibrium residual is recomputed on the selected root and checked, so a selection
    mistake would raise rather than return a plausible number.
    """
    fcm = _as_array(fcm_MPa)
    d_bottom = _as_array(c_bottom_mm)
    stiffness = beam.steel_modulus_MPa * beam.concrete_ultimate_strain
    concrete = beam.block_depth_factor * beam.thickness_mm * beam.block_stress_factor * fcm
    tension = beam.area_top_mm2 * beam.steel_yield_MPa
    yield_force = beam.area_bottom_mm2 * beam.steel_yield_MPa

    quadratic_b = beam.area_bottom_mm2 * stiffness - tension
    quadratic_c = -beam.area_bottom_mm2 * stiffness * d_bottom
    discriminant = quadratic_b**2 - 4.0 * concrete * quadratic_c
    elastic_root = (-quadratic_b + np.sqrt(discriminant)) / (2.0 * concrete)

    yielded_compression_root = (tension - yield_force) / concrete
    yielded_tension_root = (tension + yield_force) / concrete

    implied = beam.steel_modulus_MPa * beam.concrete_ultimate_strain * (
        elastic_root - d_bottom
    ) / elastic_root
    x = np.where(
        implied > beam.steel_yield_MPa,
        yielded_compression_root,
        np.where(implied < -beam.steel_yield_MPa, yielded_tension_root, elastic_root),
    )
    residual = (
        concrete * x + beam.area_bottom_mm2 * compression_steel_stress_MPa(x, d_bottom, beam)
        - tension
    )
    if np.any(np.abs(residual) > EQUILIBRIUM_TOLERANCE * tension):
        worst = float(np.max(np.abs(residual)) / tension)
        raise SectionAssumptionViolated(
            f"the solved neutral axis leaves a relative equilibrium residual of {worst:.3e}, "
            f"above the tolerance {EQUILIBRIUM_TOLERANCE:.0e}. The regime selection is wrong "
            "for at least one sample; no root is returned rather than the closest one."
        )
    return x


@dataclass(frozen=True)
class SectionState:
    """The solved hogging section at one or many samples, all arrays of the same shape."""

    neutral_axis_mm: np.ndarray
    moment_Nmm: np.ndarray
    bottom_steel_stress_MPa: np.ndarray
    tension_steel_strain: np.ndarray


def hogging_capacity(
    fcm_MPa: Any, c_bottom_mm: Any, c_top_mm: Any, beam: Beam = DEFAULT_BEAM
) -> SectionState:
    """Hogging moment capacity at the roller support, in N mm.

    Moments are taken about the tension steel, so the two contributions are the concrete block
    at ``d - lambda x / 2`` and the bottom layer at ``d - d'``. The bottom layer's contribution
    carries its own sign: where the neutral axis falls below it the layer is in tension and the
    term reduces the capacity, which is the honest arithmetic for a layer that close to the
    compression face.

    Two assumptions are checked rather than asserted in prose. The tension steel must be at or
    past yield, which is what lets its force be ``As fy``; and the neutral axis must lie above
    the tension steel, without which the section is not in the regime the formula describes.
    Both raise, because a capacity computed outside its own assumptions is exactly the kind of
    plausible number this project exists not to produce.
    """
    fcm = _as_array(fcm_MPa)
    d_bottom = _as_array(c_bottom_mm)
    d_top = _as_array(c_top_mm)
    x = neutral_axis_depth_mm(fcm, d_bottom, beam)
    if np.any(x >= d_top):
        raise SectionAssumptionViolated(
            "the neutral axis reaches the tension steel in at least one sample, so the section "
            "is not under reinforced and the rectangular block formula does not apply."
        )
    strain = beam.concrete_ultimate_strain * (d_top - x) / x
    if np.any(strain < beam.steel_yield_strain):
        worst = float(np.min(strain))
        raise SectionAssumptionViolated(
            f"the tension steel strain falls to {worst:.5f}, below the yield strain "
            f"{beam.steel_yield_strain:.5f}, so taking its force as As fy would overstate the "
            "capacity. The section is over reinforced at those samples."
        )
    stress = compression_steel_stress_MPa(x, d_bottom, beam)
    concrete_force = (
        beam.block_depth_factor * x * beam.thickness_mm * beam.block_stress_factor * fcm
    )
    moment = concrete_force * (d_top - beam.block_depth_factor * x / 2.0) + (
        beam.area_bottom_mm2 * stress * (d_top - d_bottom)
    )
    return SectionState(
        neutral_axis_mm=x,
        moment_Nmm=moment,
        bottom_steel_stress_MPa=stress,
        tension_steel_strain=strain,
    )


def peak_load_N(
    fcm_MPa: Any, c_bottom_mm: Any, c_top_mm: Any, beam: Beam = DEFAULT_BEAM
) -> np.ndarray:
    """The analytic peak load at the imposed displacement point, in N.

    The hogging capacity at the roller divided by the 800 mm lever arm to the load point. This
    is the quantity build spec 13.4 compares against the surrogate's propagated peak load.
    """
    return hogging_capacity(fcm_MPa, c_bottom_mm, c_top_mm, beam).moment_Nmm / beam.overhang_mm


def tensile_strength_MPa(fcm_MPa: Any) -> np.ndarray:
    """Mean axial tensile strength in MPa, ``0.3 (fcm - 8)**(2/3)``.

    Eurocode 2 for classes up to C50/60, written on the mean compressive strength the campaign
    varies. It is the same function of the strength the campaign's material generator used to
    scale its tension table, which is why its logarithmic elasticity of 0.933 at the mean is
    the number the cross check of build spec 13.4 ends up pointing at.
    """
    fcm = _as_array(fcm_MPa)
    if np.any(fcm <= 8.0):
        raise SectionAssumptionViolated(
            "the Eurocode 2 tensile strength expression needs a mean compressive strength "
            f"above 8 MPa; the smallest given is {float(fcm.min()):.3f} MPa."
        )
    return 0.3 * (fcm - 8.0) ** (2.0 / 3.0)


def cracking_load_N(fcm_MPa: Any, beam: Beam = DEFAULT_BEAM) -> np.ndarray:
    """The load at which the gross section first cracks in tension, in N."""
    modulus = beam.thickness_mm * beam.depth_mm**2 / 6.0
    return tensile_strength_MPa(fcm_MPa) * modulus / beam.overhang_mm


def cracked_second_moment_mm4(
    modulus_MPa: Any, c_bottom_mm: Any, c_top_mm: Any, beam: Beam = DEFAULT_BEAM
) -> np.ndarray:
    """Second moment of the fully cracked transformed section, in mm**4.

    Concrete carries compression only, both steel layers are transformed by ``n = Es / Ec``,
    and the compression zone is measured from the soffit as everywhere else in this module.
    The neutral axis solves ``b c**2 / 2 + (n - 1) As_bot (c - d') = n As_top (d - c)``, which
    is a quadratic with one positive root.
    """
    modulus = _as_array(modulus_MPa)
    d_bottom = _as_array(c_bottom_mm)
    d_top = _as_array(c_top_mm)
    ratio = beam.steel_modulus_MPa / modulus
    quadratic_a = beam.thickness_mm / 2.0
    quadratic_b = (ratio - 1.0) * beam.area_bottom_mm2 + ratio * beam.area_top_mm2
    quadratic_c = -(
        (ratio - 1.0) * beam.area_bottom_mm2 * d_bottom + ratio * beam.area_top_mm2 * d_top
    )
    depth = (-quadratic_b + np.sqrt(quadratic_b**2 - 4.0 * quadratic_a * quadratic_c)) / (
        2.0 * quadratic_a
    )
    return (
        beam.thickness_mm * depth**3 / 3.0
        + (ratio - 1.0) * beam.area_bottom_mm2 * (depth - d_bottom) ** 2
        + ratio * beam.area_top_mm2 * (d_top - depth) ** 2
    )


def tip_stiffness_N_per_mm(
    modulus_MPa: Any, second_moment_mm4: Any, beam: Beam = DEFAULT_BEAM
) -> np.ndarray:
    """Secant stiffness at the imposed displacement point, in N/mm.

    The deflection of a load at the tip of an overhang on a simply supported span is
    ``P a**2 (L + a) / (3 E I)`` in bending. The member is deep relative to its 800 mm arm, and
    the finite element model is a two dimensional continuum that carries shear deformation, so
    the shear flexibility ``a / (kappa G A)`` is added rather than dropped. It contributes
    about 3.5 percent of the total, which is small and is included anyway because leaving it
    out would be a choice nobody wrote down.

    This is the corrected version of what the salvaged script was reaching for, and it is
    stated as a stiffness rather than as a deflection under an arbitrary load.
    """
    modulus = _as_array(modulus_MPa)
    inertia = _as_array(second_moment_mm4)
    bending = beam.overhang_mm**2 * (beam.span_mm + beam.overhang_mm) / (3.0 * modulus * inertia)
    shear_modulus = modulus / (2.0 * (1.0 + beam.poisson_ratio))
    shear = beam.overhang_mm / (beam.shear_correction * shear_modulus * beam.section_area_mm2)
    return 1.0 / (bending + shear)


def log_elasticities(
    fcm_MPa: float,
    c_bottom_mm: float,
    c_top_mm: float,
    beam: Beam = DEFAULT_BEAM,
    relative_step: float = 1.0e-5,
) -> dict[str, float]:
    """``d log P / d log input`` at one point, by a central difference in log space.

    The comparison of build spec 13.4 turns on these three numbers, because a distribution can
    disagree with another for reasons that are invisible in its moments. An elasticity says how
    the model transmits each input, and the analytic model and the campaign can then be
    compared term by term instead of only in the aggregate.
    """
    base = {"Fcm_MPa": fcm_MPa, "c_nom_bottom_mm": c_bottom_mm, "c_nom_top_mm": c_top_mm}
    out: dict[str, float] = {}
    for name, value in base.items():
        moved = dict(base)
        moved[name] = value * (1.0 + relative_step)
        high = float(
            peak_load_N(moved["Fcm_MPa"], moved["c_nom_bottom_mm"], moved["c_nom_top_mm"], beam)
        )
        moved[name] = value * (1.0 - relative_step)
        low = float(
            peak_load_N(moved["Fcm_MPa"], moved["c_nom_bottom_mm"], moved["c_nom_top_mm"], beam)
        )
        out[name] = (np.log(high) - np.log(low)) / (2.0 * relative_step)
    return out


def empirical_log_elasticities(
    design: np.ndarray, response: np.ndarray, names: tuple[str, ...]
) -> dict[str, Any]:
    """Least squares elasticities of a measured response, ``log y`` on ``log x`` with a
    constant.

    A power law is not claimed to be the true response surface. It is claimed to be the
    comparison the analytic model can be held to, because the analytic model's own
    elasticities are what :func:`log_elasticities` measures, and over the range this campaign
    spans (a strength range of less than a factor of two) the two are directly comparable. The
    fit's own coefficient of determination is returned beside the coefficients, so a reader can
    see how much of the response the comparison is entitled to speak about.
    """
    x = np.asarray(design, dtype=float)
    y = np.asarray(response, dtype=float)
    if x.shape[0] != y.size or x.shape[1] != len(names):
        raise ValueError(
            f"the design {x.shape} and the response {y.shape} do not agree with the names "
            f"{names}."
        )
    matrix = np.column_stack([np.ones(x.shape[0]), np.log(x)])
    coefficients, *_ = np.linalg.lstsq(matrix, np.log(y), rcond=None)
    residual = np.log(y) - matrix @ coefficients
    return {
        "elasticities": {
            name: float(value) for name, value in zip(names, coefficients[1:])
        },
        "r2": float(1.0 - np.var(residual) / np.var(np.log(y))),
        "n": int(y.size),
    }


def cross_check(
    surrogate_sample: np.ndarray,
    analytic_sample: np.ndarray,
    model_error: float = MODEL_ERROR_FRACTION,
) -> dict[str, Any]:
    """The bracketing test of build spec 13.4, stated as two separate verdicts.

    The spec asks whether the analytic peak load distribution brackets the surrogate's within
    the stated model error. That is two questions and they can have different answers, so they
    are reported separately rather than collapsed into one boolean:

    * **central tendency**: does the analytic median lie within ``model_error`` of the
      surrogate median?
    * **dispersion**: do the analytic 5th and 95th percentiles each lie within ``model_error``
      of the surrogate's?

    The dispersion test is quantile by quantile rather than a containment of one interval in
    another widened one. Containment was the first version and it is nearly vacuous here: a
    15 percent model error on a 38 kN capacity is a 5.7 kN band, wider than the whole 5th to
    95th percentile spread of the response, so a point mass at the right place would have
    passed it. Asking each quantile to agree to the same relative tolerance is the statement
    that actually distinguishes two distributions of different width.

    Neither verdict is a threshold anyone may move after seeing the numbers. Both are recorded
    with the measurements that produced them, so a failure is diagnosed in the report rather
    than being retuned into a pass.
    """
    surrogate = np.asarray(surrogate_sample, dtype=float)
    analytic = np.asarray(analytic_sample, dtype=float)
    levels = (5.0, 50.0, 95.0)
    s_low, s_median, s_high = (float(np.percentile(surrogate, level)) for level in levels)
    a_low, a_median, a_high = (float(np.percentile(analytic, level)) for level in levels)
    ratios = {
        "p05": a_low / s_low,
        "median": a_median / s_median,
        "p95": a_high / s_high,
    }
    ratio = ratios["median"]
    return {
        "model_error": float(model_error),
        "surrogate": {
            "mean": float(surrogate.mean()),
            "std": float(surrogate.std(ddof=1)),
            "cov": float(surrogate.std(ddof=1) / surrogate.mean()),
            "p05": s_low,
            "median": s_median,
            "p95": s_high,
        },
        "analytic": {
            "mean": float(analytic.mean()),
            "std": float(analytic.std(ddof=1)),
            "cov": float(analytic.std(ddof=1) / analytic.mean()),
            "p05": a_low,
            "median": a_median,
            "p95": a_high,
        },
        "quantile_ratio": {key: float(value) for key, value in ratios.items()},
        "median_ratio": float(ratio),
        "median_relative_gap": float(ratio - 1.0),
        "central_tendency_brackets": bool(abs(ratio - 1.0) <= model_error),
        "dispersion_brackets": bool(
            abs(ratios["p05"] - 1.0) <= model_error and abs(ratios["p95"] - 1.0) <= model_error
        ),
        "dispersion_ratio": float(
            (analytic.std(ddof=1) / analytic.mean()) / (surrogate.std(ddof=1) / surrogate.mean())
        ),
    }
