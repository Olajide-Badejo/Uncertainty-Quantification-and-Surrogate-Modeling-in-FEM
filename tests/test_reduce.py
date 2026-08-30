"""Reduction contracts and the PCA properties of build spec 16.1.

The central property is that projection followed by reconstruction at full rank is the
identity. It is worth testing precisely because it is the one thing that cannot be true by
accident: a centering bug, a transposed component matrix, or a truncation off by one all
break it, and all of them would otherwise produce plausible looking scores.

The tolerance is relative rather than absolute. The amplitude block carries forces of order
1e4 N, so a 1e-8 absolute bound on a quantity of that size is asking for 12 significant
digits, which is below what a float64 SVD delivers on a matrix with that condition number.
The measured relative error is about 6e-16, machine precision, and that is what is asserted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ufem.config import load_config
from ufem.reduce import (
    BLOCKS,
    ERROR_PERCENTILES,
    error_percentiles,
    fit_basis,
    reconstruction_errors,
)

SETTINGS = settings(deadline=None, derandomize=True, max_examples=30)


def low_rank_matrix(n_rows: int, n_features: int, rank: int, seed: int) -> np.ndarray:
    """A matrix of known rank, built from a seeded generator (ground rule 13)."""
    rng = np.random.default_rng(seed)
    left = rng.normal(size=(n_rows, rank))
    right = rng.normal(size=(rank, n_features))
    return left @ right + 5.0


class TestTheIdentityProperty:
    @SETTINGS
    @given(
        n_rows=st.integers(min_value=6, max_value=30),
        n_features=st.integers(min_value=4, max_value=20),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    def test_full_rank_projection_then_reconstruction_is_the_identity(
        self, n_rows, n_features, seed
    ):
        matrix = low_rank_matrix(n_rows, n_features, min(n_rows - 1, n_features), seed)
        basis = fit_basis(matrix, "property", 1.0)
        rebuilt = basis.reconstruct(basis.project(matrix))
        scale = max(float(np.abs(matrix).max()), 1.0)
        np.testing.assert_allclose(rebuilt / scale, matrix / scale, atol=1e-10, rtol=0.0)

    def test_a_rank_deficient_family_needs_only_its_true_rank(self):
        """A rank 3 family must not be described with more than 3 components."""
        matrix = low_rank_matrix(40, 25, 3, seed=11)
        basis = fit_basis(matrix, "rank3", 0.99)
        assert basis.n_retained <= 3
        assert float(basis.explained_variance_ratio[:3].sum()) == pytest.approx(1.0, abs=1e-9)

    def test_truncated_reconstruction_error_is_zero_at_the_true_rank(self):
        matrix = low_rank_matrix(30, 18, 4, seed=5)
        basis = fit_basis(matrix, "rank4", 0.99)
        errors = reconstruction_errors(matrix, basis, 4)
        assert float(errors.max()) < 1e-12


class TestTheBasisContract:
    def test_the_sign_convention_is_pinned(self):
        """Largest magnitude entry positive, so loadings cannot flip between platforms."""
        basis = fit_basis(low_rank_matrix(25, 12, 5, seed=3), "signs", 0.99)
        for component in basis.components:
            assert component[int(np.argmax(np.abs(component)))] > 0.0

    def test_components_are_orthonormal(self):
        basis = fit_basis(low_rank_matrix(30, 15, 6, seed=7), "ortho", 0.99)
        retained = basis.components[: basis.n_retained]
        gram = retained @ retained.T
        np.testing.assert_allclose(gram, np.eye(retained.shape[0]), atol=1e-10, rtol=0.0)

    def test_explained_variance_ratios_sum_to_one(self):
        basis = fit_basis(low_rank_matrix(30, 15, 6, seed=9), "ratios", 0.99)
        assert float(basis.explained_variance_ratio.sum()) == pytest.approx(1.0, abs=1e-12)

    def test_more_components_never_increase_the_error(self):
        """Monotonicity in the truncation: an extra component cannot hurt."""
        matrix = low_rank_matrix(35, 20, 8, seed=13)
        basis = fit_basis(matrix, "monotone", 0.99)
        means = [
            float(reconstruction_errors(matrix, basis, k).mean()) for k in range(1, 9)
        ]
        assert all(later <= earlier + 1e-12 for earlier, later in zip(means, means[1:]))

    def test_a_constant_family_is_rejected(self):
        with pytest.raises(ValueError, match="zero total variance"):
            fit_basis(np.ones((10, 5)), "constant", 0.99)

    def test_a_single_row_is_rejected(self):
        with pytest.raises(ValueError, match="too few to estimate"):
            fit_basis(np.ones((1, 5)), "single", 0.99)

    def test_projecting_the_wrong_width_raises(self):
        basis = fit_basis(low_rank_matrix(20, 10, 4, seed=17), "width", 0.99)
        with pytest.raises(ValueError, match="expects rows of 10 features"):
            basis.project(np.zeros((3, 7)))


class TestErrorPercentiles:
    def test_it_reports_the_documented_percentiles(self):
        keys = set(error_percentiles(np.linspace(0.0, 1.0, 101)))
        assert keys == {f"p{value}" for value in ERROR_PERCENTILES}

    def test_the_percentiles_are_ordered(self):
        reported = error_percentiles(np.random.default_rng(0).random(500))
        assert reported["p50"] <= reported["p90"] <= reported["p99"]


@pytest.mark.fullstack
class TestAgainstTheArtifactStore:
    @staticmethod
    @pytest.fixture(scope="class")
    def artifacts(repo_root):
        import json

        from ufem.config import config_hash
        from ufem.manifest import stage_dir
        from ufem.reduce import BASIS_JSON, RECONSTRUCTION_JSON, SCORES_PARQUET, STAGE_NAME

        config = load_config(repo_root)
        directory = stage_dir(
            repo_root / config.pipeline.paths.artifact_root, STAGE_NAME, config_hash(config)
        )
        if not (directory / "manifest.json").is_file():
            pytest.skip(f"the reduce stage has not run for this config hash: {directory}")
        return {
            "bases": json.loads((directory / BASIS_JSON).read_text(encoding="utf-8")),
            "errors": json.loads((directory / RECONSTRUCTION_JSON).read_text(encoding="utf-8")),
            "scores": pd.read_parquet(directory / SCORES_PARQUET),
        }

    def test_all_three_blocks_are_present(self, artifacts):
        assert {block["name"] for block in artifacts["bases"]["blocks"]} == set(BLOCKS)

    def test_every_block_reaches_its_variance_target(self, artifacts):
        target = artifacts["bases"]["variance_target"]
        for block in artifacts["bases"]["blocks"]:
            assert block["variance_explained_by_retained"] >= target

    def test_the_scores_table_has_one_column_per_retained_component(self, artifacts):
        scores, columns = artifacts["scores"], set(artifacts["scores"].columns)
        assert len(scores) == 198
        for block in artifacts["bases"]["blocks"]:
            for index in range(int(block["n_retained"])):
                assert f"{block['name']}_pc{index + 1}" in columns

    def test_the_persisted_percentiles_are_ordered_and_finite(self, artifacts):
        for name, record in artifacts["errors"].items():
            assert record["p50"] <= record["p90"] <= record["p99"], name
            assert np.isfinite([record["p50"], record["p90"], record["p99"]]).all(), name

    def test_the_amplitude_block_is_low_rank_as_the_spec_expects(self, artifacts):
        """Build spec 10.2 expects 3 to 6 components for the registered amplitude."""
        amplitude = next(
            block for block in artifacts["bases"]["blocks"] if block["name"] == "amplitude"
        )
        assert 3 <= int(amplitude["n_retained"]) <= 6
