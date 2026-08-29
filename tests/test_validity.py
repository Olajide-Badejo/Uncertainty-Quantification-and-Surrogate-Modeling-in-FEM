"""The validity domain contract of build spec 9.4.

Binding law 4 says every downstream product either models the censoring or carries a machine
checked validity domain that excludes the censored corner. This file is the machine check.

Two properties matter and both are tested against the real fitted domain. A point deep in the
region where the campaign succeeded must be inside. A point in the censored corner, low top
cover with high strength, which the audit measured failing at 76 percent in the lowest top
cover quartile, must be outside. A contract that answered True everywhere would pass a test
that only ever asked about good points, so the corner case is the one that carries the weight.

The failure path is tested too. Ground rule 8 forbids a silent fallback, so a missing audit
artifact must raise a named diagnostic rather than quietly returning an open domain.
"""

from __future__ import annotations

import numpy as np
import pytest

from ufem.audit import VALIDITY_DOMAIN_JSON
from ufem.config import FEATURE_ORDER, load_config
from ufem.validity import (
    ValidityDomainUnavailable,
    in_validity_domain,
    load_validity_domain,
)

#: Deep inside the surviving region: median strength, median bottom cover, and a top cover
#: in the third quartile, which the audit measured as the lowest failure rate of the four.
INSIDE_POINT = [27.5, 27.0, 226.0]

#: The censored corner build spec 5.5 names: low top cover, high strength. The lowest top
#: cover quartile failed at 76 percent and the highest strength quartile at 63 percent, so a
#: prediction here is an extrapolation into a region the solver mostly could not complete.
CENSORED_CORNER = [35.0, 27.0, 213.0]


@pytest.fixture(scope="module")
def domain(repo_root):
    """The real fitted domain, or a skip naming why it is unavailable.

    Every test in :class:`TestTheContract` must depend on this fixture, including the ones
    that go on to call :func:`in_validity_domain` rather than touching the returned object.
    The artifact store is gitignored, so on a clean checkout (CI, or a fresh clone) the audit
    stage has not run and there is nothing to assert against. A test that called
    ``in_validity_domain`` without taking this fixture bypassed the skip and turned a missing
    artifact into six red tests, which is what happened on the P2 merge: the absence of an
    artifact is not the same thing as a broken contract, and only one of the two should be
    able to fail this suite.
    """
    try:
        return load_validity_domain(repo_root)
    except ValidityDomainUnavailable as err:
        pytest.skip(f"the audit stage has not run: {err}")


@pytest.mark.fullstack
class TestTheContract:
    def test_a_point_deep_inside_the_surviving_region_is_inside(self, repo_root, domain):
        assert bool(in_validity_domain(np.array([INSIDE_POINT]), repo_root)[0])

    def test_the_censored_corner_is_outside(self, repo_root, domain):
        """Low top cover and high strength: the corner binding law 4 exists to exclude."""
        assert not bool(in_validity_domain(np.array([CENSORED_CORNER]), repo_root)[0])

    def test_the_corner_is_excluded_by_probability_not_only_by_the_box(
        self, repo_root, domain
    ):
        """The corner sits inside the design box, so the model is what rejects it.

        This distinction matters. If the corner were merely outside the design bounds, the
        test above would pass without the completion model contributing anything, and the
        contract would collapse to a bounding box check.
        """
        corner = np.array([CENSORED_CORNER])
        assert bool(domain.inside_design_box(corner)[0])
        assert float(domain.completion_probability(corner)[0]) < domain.threshold

    def test_a_point_far_outside_the_design_is_outside(self, repo_root, domain):
        """Design density: no data under a prediction means no prediction."""
        far = np.array([[100.0, 27.0, 223.0]])
        assert not bool(in_validity_domain(far, repo_root)[0])

    def test_it_answers_a_batch_row_by_row(self, repo_root, domain):
        batch = np.array([INSIDE_POINT, CENSORED_CORNER, INSIDE_POINT])
        answer = in_validity_domain(batch, repo_root)
        assert answer.shape == (3,)
        assert answer.dtype == bool
        assert list(answer) == [True, False, True]

    def test_it_accepts_a_frame_through_the_feature_contract(self, repo_root, domain):
        import pandas as pd

        frame = pd.DataFrame([dict(zip(FEATURE_ORDER, INSIDE_POINT))])
        assert bool(in_validity_domain(frame, repo_root)[0])

    def test_a_matrix_of_the_wrong_width_raises_naming_the_contract(self, repo_root, domain):
        with pytest.raises(ValueError, match="in that order"):
            in_validity_domain(np.zeros((2, 4)), repo_root)

    def test_the_domain_excludes_more_failed_jobs_than_valid_ones(self, domain):
        """The domain must actually track the censoring, not merely exist."""
        record = domain.record
        assert record["valid_jobs_inside_fraction"] > record["failed_jobs_inside_fraction"]

    def test_the_stamped_model_digest_is_rechecked_on_load(self, repo_root, domain):
        """Binding law 5: the domain is only usable with the model it was stamped on."""
        assert len(domain.record["model_sha256"]) == 64
        assert domain.record["feature_order"] == list(FEATURE_ORDER)


class TestTheFailurePath:
    def test_an_unrun_audit_raises_a_named_diagnostic(self, tmp_path, repo_root):
        """Ground rule 8: no silent fallback to an open domain."""
        config = load_config(repo_root)
        # A repo root whose artifact store is empty: the config loads, the stage has not run.
        for name in ("configs", "docs"):
            (tmp_path / name).mkdir()
        for relative in (
            "configs/pipeline.yaml",
            "configs/probabilistic_model.yaml",
        ):
            (tmp_path / relative).write_text(
                (repo_root / relative).read_text(encoding="utf-8"), encoding="utf-8"
            )
        with pytest.raises(ValidityDomainUnavailable) as caught:
            load_validity_domain(tmp_path, config)
        message = str(caught.value)
        assert VALIDITY_DOMAIN_JSON in message
        assert "ufem run audit" in message
