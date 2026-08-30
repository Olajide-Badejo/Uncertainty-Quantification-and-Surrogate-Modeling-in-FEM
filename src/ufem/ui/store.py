"""The artifact store, loaded once and handed to the panels.

Binding law 5 in one object. Every panel of UFEM Lab reads from a :class:`LabStore`, and a
:class:`LabStore` is built only out of files a stage wrote under the current config hash with a
manifest beside them. There is no other source of a number in this package.

Ground rule 8: nothing here falls back. A missing stage raises :class:`LabArtifactMissing`
naming the stage and the command that produces it, because a dashboard that opened with an
empty panel would be telling its reader that the panel has nothing to say, when what happened
is that the pipeline has not run.

The load is eager and it is slow, about two seconds, dominated by rebuilding the Gaussian
process modules from their stored parameters. That cost is paid once at startup rather than
per interaction, which is what leaves the slider inside the latency budget of build spec 15.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ufem.audit import (
    CENSORING_JSON,
    COMPLETION_JSON,
    STATUS_VALID,
    VALIDITY_DOMAIN_JSON,
    VALIDITY_PARQUET,
)
from ufem.audit import STAGE_NAME as AUDIT_STAGE
from ufem.calibrate import (
    BAND_EXAMPLES_PARQUET,
    CALIBRATION_JSON,
    COVERAGE_SWEEP_PARQUET,
    SCALAR_CONFORMAL_PARQUET,
    SIGNAL_DAMAGE,
    SIGNAL_FORCE,
)
from ufem.calibrate import STAGE_NAME as CALIBRATE_STAGE
from ufem.config import FEATURE_ORDER, Config, config_hash, load_config
from ufem.grid import DAMAGE_GRID_PARQUET, QOI_PARQUET, RF2_GRID_PARQUET
from ufem.grid import STAGE_NAME as GRID_STAGE
from ufem.ingest import DESIGN_PARQUET
from ufem.ingest import STAGE_NAME as INGEST_STAGE
from ufem.manifest import load_manifest, stage_dir
from ufem.propagate import (
    DENSITY_PARQUET,
    MC_SUBSAMPLE_PARQUET,
    PROPAGATION_JSON,
    RELIABILITY_PARQUET,
)
from ufem.propagate import STAGE_NAME as PROPAGATE_STAGE
from ufem.sensitivity import (
    AGREEMENT_PARQUET,
    FUNCTIONAL_INDICES_PARQUET,
    GP_INDICES_PARQUET,
    PCE_INDICES_PARQUET,
    SENSITIVITY_JSON,
)
from ufem.sensitivity import STAGE_NAME as SENSITIVITY_STAGE
from ufem.surrogate import STAGE_NAME as SURROGATE_STAGE
from ufem.surrogate import SURROGATE_JSON, SurrogateModel, configure_torch
from ufem.ui import layout
from ufem.validate import BASELINES_JSON
from ufem.validate import STAGE_NAME as VALIDATE_STAGE
from ufem.validity import ValidityDomain, load_validity_domain

#: The stages the dashboard reads, in pipeline order, with the command that produces each. A
#: stage missing from the store is reported with its own command rather than with `ufem run
#: all`, because rerunning everything to recover one stage is how a cache stops being useful.
REQUIRED_STAGES: tuple[tuple[str, str], ...] = (
    (INGEST_STAGE, "ufem run ingest"),
    (GRID_STAGE, "ufem run grid"),
    (AUDIT_STAGE, "ufem run audit"),
    (SURROGATE_STAGE, "ufem run surrogate"),
    (VALIDATE_STAGE, "ufem run validate"),
    (CALIBRATE_STAGE, "ufem run calibrate"),
    (SENSITIVITY_STAGE, "ufem run sensitivity"),
    (PROPAGATE_STAGE, "ufem run propagate"),
)

#: Display names for the three inputs. Names, not numbers: the values, the bounds and the units
#: all come from the artifacts.
INPUT_LABELS: dict[str, tuple[str, str]] = {
    "Fcm_MPa": ("Mean compressive strength", "MPa"),
    "c_nom_bottom_mm": ("Bottom cover", "mm"),
    "c_nom_top_mm": ("Top cover", "mm"),
}


class LabArtifactMissing(RuntimeError):
    """A stage UFEM Lab reads from has not run, and no panel will be filled with a guess."""


def _read_json(path: Path, role: str, how: str) -> dict[str, Any]:
    if not path.is_file():
        raise LabArtifactMissing(
            f"UFEM Lab needs the {role} at {path}, which does not exist. Run `{how}` for this "
            "config hash. The dashboard reads artifacts and computes no substitute for one."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _read_parquet(path: Path, role: str, how: str) -> pd.DataFrame:
    if not path.is_file():
        raise LabArtifactMissing(
            f"UFEM Lab needs the {role} at {path}, which does not exist. Run `{how}` for this "
            "config hash."
        )
    return pd.read_parquet(path)


@dataclass(frozen=True)
class LabStore:
    """Everything the five panels read, loaded once from one config hash."""

    repo_root: Path
    config: Config
    config_sha256: str
    artifact_root: Path
    surrogate: SurrogateModel
    domain: ValidityDomain
    calibration: dict[str, Any]
    conformal: pd.DataFrame
    band_examples: pd.DataFrame
    coverage_sweep: pd.DataFrame
    design: pd.DataFrame
    validity: pd.DataFrame
    qoi: pd.DataFrame
    force_grid: pd.DataFrame
    damage_grid: pd.DataFrame
    censoring: dict[str, Any]
    completion: dict[str, Any]
    domain_record: dict[str, Any]
    sensitivity: dict[str, Any]
    pce_indices: pd.DataFrame
    gp_indices: pd.DataFrame
    agreement: pd.DataFrame
    functional_indices: pd.DataFrame
    propagation: dict[str, Any]
    reliability: pd.DataFrame
    density: pd.DataFrame
    mc_subsample: pd.DataFrame
    baselines: dict[str, Any]
    surrogate_record: dict[str, Any]
    manifests: dict[str, dict[str, Any]]

    # -- construction -------------------------------------------------------

    @classmethod
    def load(cls, repo_root: Path | str, config: Config | None = None) -> LabStore:
        """Build the store, or raise naming the first stage that has not run."""
        root = Path(repo_root).resolve()
        resolved = config if config is not None else load_config(root)
        digest = config_hash(resolved)
        artifact_root = root / resolved.pipeline.paths.artifact_root
        directories: dict[str, Path] = {}
        manifests: dict[str, dict[str, Any]] = {}
        for stage, how in REQUIRED_STAGES:
            directory = stage_dir(artifact_root, stage, digest)
            if not (directory / "manifest.json").is_file():
                raise LabArtifactMissing(
                    f"the {stage} stage has no manifest at {directory}, so UFEM Lab has "
                    f"nothing to show for it. Run `{how}` for config "
                    f"{digest[: layout.SHORT_HASH_CHARS]}."
                )
            directories[stage] = directory
            manifests[stage] = load_manifest(directory)

        configure_torch()
        ingest = directories[INGEST_STAGE]
        grid = directories[GRID_STAGE]
        audit = directories[AUDIT_STAGE]
        calibrate = directories[CALIBRATE_STAGE]
        sensitivity = directories[SENSITIVITY_STAGE]
        propagate = directories[PROPAGATE_STAGE]
        validate = directories[VALIDATE_STAGE]
        surrogate_dir = directories[SURROGATE_STAGE]

        propagation = _read_json(
            propagate / PROPAGATION_JSON, "propagation result", "ufem run propagate"
        )
        if "subsample" not in propagation:
            raise LabArtifactMissing(
                f"the propagation result at {propagate / PROPAGATION_JSON} predates the "
                "persisted Monte Carlo subsample, so the reliability panel's threshold slider "
                "has no rows to recount. Run `ufem run propagate --force`."
            )
        return cls(
            repo_root=root,
            config=resolved,
            config_sha256=digest,
            artifact_root=artifact_root,
            surrogate=SurrogateModel.load(artifact_root, digest),
            domain=load_validity_domain(root, resolved),
            calibration=_read_json(
                calibrate / CALIBRATION_JSON, "calibration result", "ufem run calibrate"
            ),
            conformal=_read_parquet(
                calibrate / SCALAR_CONFORMAL_PARQUET,
                "scalar conformal scores",
                "ufem run calibrate",
            ),
            band_examples=_read_parquet(
                calibrate / BAND_EXAMPLES_PARQUET, "band examples", "ufem run calibrate"
            ),
            coverage_sweep=_read_parquet(
                calibrate / COVERAGE_SWEEP_PARQUET, "coverage sweep", "ufem run calibrate"
            ),
            design=_read_parquet(ingest / DESIGN_PARQUET, "LHS design", "ufem run ingest"),
            validity=_read_parquet(
                audit / VALIDITY_PARQUET, "validity classification", "ufem run audit"
            ),
            qoi=_read_parquet(grid / QOI_PARQUET, "QoI table", "ufem run grid"),
            force_grid=_read_parquet(
                grid / RF2_GRID_PARQUET, "load displacement grid", "ufem run grid"
            ),
            damage_grid=_read_parquet(
                grid / DAMAGE_GRID_PARQUET, "damage grid", "ufem run grid"
            ),
            censoring=_read_json(
                audit / CENSORING_JSON, "censoring statistics", "ufem run audit"
            ),
            completion=_read_json(
                audit / COMPLETION_JSON, "completion model report", "ufem run audit"
            ),
            domain_record=_read_json(
                audit / VALIDITY_DOMAIN_JSON, "validity domain", "ufem run audit"
            ),
            sensitivity=_read_json(
                sensitivity / SENSITIVITY_JSON, "sensitivity result", "ufem run sensitivity"
            ),
            pce_indices=_read_parquet(
                sensitivity / PCE_INDICES_PARQUET, "chaos indices", "ufem run sensitivity"
            ),
            gp_indices=_read_parquet(
                sensitivity / GP_INDICES_PARQUET,
                "posterior Sobol indices",
                "ufem run sensitivity",
            ),
            agreement=_read_parquet(
                sensitivity / AGREEMENT_PARQUET,
                "chaos against posterior agreement",
                "ufem run sensitivity",
            ),
            functional_indices=_read_parquet(
                sensitivity / FUNCTIONAL_INDICES_PARQUET,
                "functional indices",
                "ufem run sensitivity",
            ),
            propagation=propagation,
            reliability=_read_parquet(
                propagate / RELIABILITY_PARQUET, "reliability table", "ufem run propagate"
            ),
            density=_read_parquet(
                propagate / DENSITY_PARQUET, "QoI densities", "ufem run propagate"
            ),
            mc_subsample=_read_parquet(
                propagate / MC_SUBSAMPLE_PARQUET,
                "persisted Monte Carlo subsample",
                "ufem run propagate --force",
            ),
            baselines=_read_json(
                validate / BASELINES_JSON, "baselines and validation", "ufem run validate"
            ),
            surrogate_record=_read_json(
                surrogate_dir / SURROGATE_JSON, "surrogate record", "ufem run surrogate"
            ),
            manifests=manifests,
        )

    # -- what the panels ask for -------------------------------------------

    @cached_property
    def posterior_pieces(self) -> dict[str, Any]:
        """The query independent part of every fitted posterior, assembled once.

        This is what makes the predict panel fast enough to meet the latency budget of build
        spec 15: the expensive part of a Gaussian process prediction does not depend on where
        it is being asked, so it is done at startup and reused on every slider move.
        """
        from ufem.propagate import posterior_pieces

        return {name: posterior_pieces(gp) for name, gp in self.surrogate.models.items()}

    @cached_property
    def conformal_scores(self) -> dict[str, np.ndarray]:
        """The jackknife+ scores per scalar target, as the calibration stage measured them."""
        return {
            name: self.conformal.loc[self.conformal["target"] == name, "score"].to_numpy(
                dtype=float
            )
            for name in self.surrogate.scalar_targets
        }

    @cached_property
    def scalar_scaling(self) -> dict[str, float]:
        """The variance scaling factor per scalar target, read from the calibration."""
        return {
            name: float(self.calibration["scalar"][name]["variance_scaling_factor"])
            for name in self.surrogate.scalar_targets
        }

    @property
    def band_alpha(self) -> float:
        """The band level every calibrated interval in the dashboard is drawn at.

        Read from the propagation context rather than restated, so the dashboard and the
        reliability table are quoting the same level by construction.
        """
        return float(self.propagation["context"]["band_alpha"])

    def functional_band(self, signal: str) -> tuple[float, float]:
        """``(band_scale, variance_scaling_factor)`` for one curve family at the band level.

        The deployed simultaneous band of build spec 11.2 is
        ``mean +/- band_scale * variance_scaling_factor * sigma(u)``, and both factors are
        measurements the calibration stage made against held out curves.
        """
        if signal not in (SIGNAL_FORCE, SIGNAL_DAMAGE):
            raise KeyError(
                f"the calibration artifact carries the signals {SIGNAL_FORCE!r} and "
                f"{SIGNAL_DAMAGE!r}; asked for {signal!r}."
            )
        record = self.calibration["functional"][signal]
        key = f"{self.band_alpha:g}"
        if key not in record["bands"]:
            raise KeyError(
                f"the calibration measured no {signal} band at alpha {key}; it carries "
                f"{sorted(record['bands'])}. The dashboard draws only levels whose coverage "
                "was measured."
            )
        return (
            float(record["bands"][key]["band_scale"]),
            float(record["variance_scaling_factor"]),
        )

    @property
    def design_bounds(self) -> dict[str, tuple[float, float]]:
        """The executed design's box, per input, as the validity domain artifact stamped it.

        The sliders are bounded to this and not to the marginal distributions: outside the box
        there are no runs, so a slider that could leave it would be inviting an extrapolation
        the completion model has no evidence for (build spec 9.4).
        """
        bounds = self.domain_record["design_bounds"]
        return {name: (float(bounds[name][0]), float(bounds[name][1])) for name in FEATURE_ORDER}

    @property
    def design_midpoints(self) -> dict[str, float]:
        """The median of each input over the executed design, where the sliders start."""
        return {
            name: float(np.median(self.design[name].to_numpy(dtype=float)))
            for name in FEATURE_ORDER
        }

    @cached_property
    def design_with_status(self) -> pd.DataFrame:
        """The 400 point design with its completion status and its completion probability.

        The status comes from the audit classification and the probability from the fitted
        completion model, so the scatter matrix and the graying rule are reading one answer.
        """
        frame = self.validity[list(FEATURE_ORDER) + ["sample_id", "job", "status"]].copy()
        frame["completed"] = frame["status"] == STATUS_VALID
        matrix = frame[list(FEATURE_ORDER)].to_numpy(dtype=float)
        frame["completion_probability"] = self.domain.completion_probability(matrix)
        frame["inside_domain"] = self.domain.inside_design_box(matrix) & (
            frame["completion_probability"].to_numpy(dtype=float) >= self.domain.threshold
        )
        return frame

    @cached_property
    def u_grid(self) -> np.ndarray:
        """The common displacement grid, in mm, as the surrogate's basis holds it."""
        return np.asarray(self.surrogate.basis.u_grid, dtype=float)

    def observed_curves(self, job: str) -> tuple[np.ndarray, np.ndarray]:
        """The finite element load displacement and damage curves for one completed job."""
        force = self.force_grid.loc[self.force_grid["job"] == job]
        damage = self.damage_grid.loc[self.damage_grid["job"] == job]
        if force.empty or damage.empty:
            raise KeyError(
                f"no gridded curve for job {job!r}. The grid stage carries only the completed "
                f"runs, of which there are {len(self.force_grid)}."
            )
        columns = [name for name in self.force_grid.columns if name != "job"]
        return (
            force[columns].to_numpy(dtype=float).ravel(),
            damage[columns].to_numpy(dtype=float).ravel(),
        )

    def job_inputs(self, job: str) -> dict[str, float]:
        """The three input values of one job, from the audit classification table."""
        row = self.validity.loc[self.validity["job"] == job]
        if row.empty:
            raise KeyError(f"no design row for job {job!r} in the validity classification.")
        return {name: float(row.iloc[0][name]) for name in FEATURE_ORDER}

    @cached_property
    def enriched_corners(self) -> list[dict[str, Any]]:
        """Where the campaign failed, per input, from the censoring statistics.

        One entry per input whose association with completion is significant at the configured
        level, carrying the quantile bin with the highest failure rate and that rate. This is
        what lets the predict panel name the censored corner instead of saying that a point is
        outside a domain and leaving the reader to wonder which one.
        """
        corners = []
        for name in FEATURE_ORDER:
            record = self.censoring["by_input"][name]
            if not record["significant_at_level"]:
                continue
            worst = max(record["quantile_failure_rates"], key=lambda bin_: bin_["fail_rate"])
            corners.append(
                {
                    "input": name,
                    "label": INPUT_LABELS[name][0],
                    "unit": INPUT_LABELS[name][1],
                    "bin": worst["bin"],
                    "low": float(worst["low"]),
                    "high": float(worst["high"]),
                    "fail_rate": float(worst["fail_rate"]),
                    "n": int(worst["n"]),
                    "n_failed": int(worst["n_failed"]),
                    "chi2_p_value": float(record["chi2_p_value"]),
                }
            )
        return corners

    @cached_property
    def limit_states(self) -> list[dict[str, Any]]:
        """The declared limit states with everything the propagate stage measured about each."""
        return list(self.propagation["limit_states"])

    def subsample_limit_state(self, config_field: str) -> dict[str, Any]:
        """What the persisted rows say about one limit state at its configured threshold."""
        records = self.propagation["subsample"]["limit_states"]
        if config_field not in records:
            raise KeyError(
                f"no persisted limit state {config_field!r}; the artifact carries "
                f"{sorted(records)}."
            )
        return records[config_field]

    @property
    def package_versions(self) -> dict[str, str]:
        """The resolved stack the surrogate stage recorded, which is the model's provenance."""
        return dict(self.manifests[SURROGATE_STAGE]["packages"])

    @property
    def git_state(self) -> dict[str, Any]:
        """The commit the surrogate artifact was produced at, and whether the tree was dirty."""
        return dict(self.manifests[SURROGATE_STAGE]["git"])
