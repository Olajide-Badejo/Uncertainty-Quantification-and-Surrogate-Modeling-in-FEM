"""Contract tests over the single source of truth config.

These run against the real ``configs/*.yaml``, not a fixture copy, because the thing worth
pinning is the config the pipeline actually loads.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from ufem.config import (
    FEATURE_ORDER,
    Config,
    ProbabilisticModel,
    config_hash,
    derived_E,
    features,
    input_distributions,
    load_config,
)


@pytest.fixture(scope="module")
def config(repo_root):
    return load_config(repo_root)


@pytest.fixture(scope="module")
def raw_model(repo_root):
    text = (repo_root / "configs" / "probabilistic_model.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def test_feature_order_is_the_contract(config):
    """Build spec 9.2: exactly three features, in this order, and E is not one of them."""
    assert tuple(config.probabilistic_model.feature_order) == FEATURE_ORDER
    assert FEATURE_ORDER == ("Fcm_MPa", "c_nom_bottom_mm", "c_nom_top_mm")
    assert "E_MPa" not in config.probabilistic_model.feature_order


def test_features_builds_the_design_matrix_in_order():
    df = pd.DataFrame(
        {
            "c_nom_top_mm": [223.0, 220.0],
            "Fcm_MPa": [28.0, 30.0],
            "irrelevant": [0, 0],
            "c_nom_bottom_mm": [27.0, 25.0],
        }
    )
    matrix = features(df)
    assert matrix.shape == (2, 3)
    # Column order follows the contract, not the frame's own column order.
    np.testing.assert_allclose(matrix[0], [28.0, 27.0, 223.0], atol=0.0)
    np.testing.assert_allclose(matrix[1], [30.0, 25.0, 220.0], atol=0.0)


def test_features_raises_named_keyerror_on_missing_column():
    df = pd.DataFrame({"Fcm_MPa": [28.0], "c_nom_bottom_mm": [27.0]})
    with pytest.raises(KeyError) as excinfo:
        features(df)
    assert "c_nom_top_mm" in str(excinfo.value)


def test_derived_E_matches_eurocode_expression():
    """E = 22000 * (Fcm/10)**0.3, the expression the audit proved exact to 2e-15."""
    assert derived_E(28.0) == pytest.approx(22000.0 * (2.8**0.3), abs=1e-9)
    values = np.array([20.0, 28.0, 38.0])
    np.testing.assert_allclose(derived_E(values), 22000.0 * (values / 10.0) ** 0.3, atol=1e-9)


def test_lognormal_reproduces_declared_mean_and_cov(config):
    """The lognormal is parameterized so the variable itself has mean 28 and CoV 0.10."""
    fcm = input_distributions(config)["Fcm_MPa"]
    assert fcm.mean() == pytest.approx(28.0, abs=1e-9)
    assert fcm.std() / fcm.mean() == pytest.approx(0.10, abs=1e-9)

    declared = config.probabilistic_model.variables["Fcm_MPa"]
    mu_ln, sigma_ln = declared.log_params()
    assert sigma_ln == pytest.approx(math.sqrt(math.log(1.0 + 0.10**2)), abs=1e-12)
    assert mu_ln == pytest.approx(math.log(28.0) - sigma_ln**2 / 2.0, abs=1e-12)


def test_normal_inputs_match_declared_parameters(config):
    dists = input_distributions(config)
    assert dists["c_nom_bottom_mm"].mean() == pytest.approx(27.0, abs=1e-12)
    assert dists["c_nom_bottom_mm"].std() == pytest.approx(3.0, abs=1e-12)
    assert dists["c_nom_top_mm"].mean() == pytest.approx(223.0, abs=1e-12)
    assert dists["c_nom_top_mm"].std() == pytest.approx(5.0, abs=1e-12)


def test_input_distributions_follow_feature_order(config):
    assert tuple(input_distributions(config)) == FEATURE_ORDER


def test_config_hash_is_stable_across_two_loads(repo_root):
    assert config_hash(load_config(repo_root)) == config_hash(load_config(repo_root))


def test_config_hash_changes_when_a_parameter_changes(config):
    """Bump one sigma and the run identity must move."""
    payload = config.model_dump(mode="json")
    payload["probabilistic_model"]["variables"]["c_nom_top_mm"]["sigma"] = 5.5
    assert config_hash(Config(**payload)) != config_hash(config)


def test_config_hash_changes_on_a_pipeline_parameter(config):
    payload = config.model_dump(mode="json")
    payload["pipeline"]["grid"]["n_points"] = 401
    assert config_hash(Config(**payload)) != config_hash(config)


def test_geometry_with_cover_outside_the_section_is_rejected(raw_model):
    """A top cover mean of 260 mm does not fit in a 250 mm section."""
    broken = copy.deepcopy(raw_model)
    broken["variables"]["c_nom_top_mm"]["mu"] = 260.0
    with pytest.raises(ValidationError, match="section depth"):
        ProbabilisticModel(**broken)


def test_negative_cover_mean_is_rejected(raw_model):
    broken = copy.deepcopy(raw_model)
    broken["variables"]["c_nom_bottom_mm"]["mu"] = -5.0
    with pytest.raises(ValidationError, match="section depth"):
        ProbabilisticModel(**broken)


def test_inverted_cover_order_is_rejected(raw_model):
    """Both covers are measured from the soffit, so bottom below top is not optional."""
    broken = copy.deepcopy(raw_model)
    broken["variables"]["c_nom_bottom_mm"]["mu"] = 230.0
    with pytest.raises(ValidationError, match="must sit lower"):
        ProbabilisticModel(**broken)


@pytest.mark.parametrize("sigma", [0.0, -3.0])
def test_non_positive_sigma_is_rejected(raw_model, sigma):
    broken = copy.deepcopy(raw_model)
    broken["variables"]["c_nom_bottom_mm"]["sigma"] = sigma
    with pytest.raises(ValidationError):
        ProbabilisticModel(**broken)


@pytest.mark.parametrize("cov", [0.0, -0.10])
def test_non_positive_cov_is_rejected(raw_model, cov):
    broken = copy.deepcopy(raw_model)
    broken["variables"]["Fcm_MPa"]["cov"] = cov
    with pytest.raises(ValidationError):
        ProbabilisticModel(**broken)


def test_E_as_a_feature_is_rejected(raw_model):
    """Binding law: E and Fcm are one random variable, so E can never be a column."""
    broken = copy.deepcopy(raw_model)
    broken["feature_order"] = ["Fcm_MPa", "c_nom_bottom_mm", "c_nom_top_mm", "E_MPa"]
    with pytest.raises(ValidationError, match="E_MPa"):
        ProbabilisticModel(**broken)


def test_reordered_features_are_rejected(raw_model):
    broken = copy.deepcopy(raw_model)
    broken["feature_order"] = ["c_nom_bottom_mm", "Fcm_MPa", "c_nom_top_mm"]
    with pytest.raises(ValidationError, match="pinned contract"):
        ProbabilisticModel(**broken)


def test_unknown_config_key_is_rejected(raw_model):
    """A misspelled key must fail loudly rather than being silently ignored."""
    broken = copy.deepcopy(raw_model)
    broken["varaibles"] = {}
    with pytest.raises(ValidationError):
        ProbabilisticModel(**broken)


def test_pipeline_values_are_the_declared_ones(config):
    pipeline = config.pipeline
    assert (pipeline.grid.u_min_mm, pipeline.grid.u_max_mm) == (0.0, 20.0)
    assert pipeline.grid.n_points == 201
    assert pipeline.pca.variance_target == 0.99
    assert pipeline.kernel.nu == 2.5
    assert pipeline.kernel.ard is True
    assert pipeline.mc.n_samples == 100000
    assert pipeline.conformal.alphas == [0.1, 0.05]
