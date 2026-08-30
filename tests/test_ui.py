"""UFEM Lab: the binding law, the graying rule, the recount, the export, and the budget.

Build spec section 15 and phase P8. Five things are worth pinning about a dashboard, and the
first one is the only one that is about the dashboard being a dashboard:

1. **Binding law 5.** No computed constant anywhere in ``src/ufem/ui/``. The check is parsed
   rather than grepped and it is planted against, because a law test that has never fired on a
   violation is a law test nobody has checked.
2. **The graying rule.** A prediction outside the validity domain is marked as such and the
   warning names the censored corner, from the audit artifact rather than from a sentence
   somebody wrote into the UI.
3. **The recount.** The reliability panel's threshold slider is
   :func:`ufem.propagate.recompute_limit_state` over persisted rows, and at the configured
   threshold it reproduces the artifact's own numbers exactly.
4. **The export.** What the export button writes round trips through JSON and carries the
   hashes that make it traceable.
5. **The budget.** Predict plus figure plus serialization under 50 ms median over a hundred
   seeded slider positions, which is the server side half of build spec 15's latency budget.
   The browser half is the playwright test at the bottom, marked slow and skipped wherever the
   browser is not installed.

Everything that needs the fitted surrogate carries the ``fullstack`` marker and skips when the
artifact store is empty, so the light stack CI job runs the parts that are pure Python and
skips the parts that are not, which is the lesson of docs/DEFECT_LOG.md applied again.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest

import dash_lint


@pytest.fixture(scope="module")
def store(repo_root):
    """The loaded artifact store, or a skip naming what has not run."""
    pytest.importorskip("torch")
    pytest.importorskip("plotly")
    from ufem.ui.store import LabArtifactMissing, LabStore

    try:
        return LabStore.load(repo_root)
    except LabArtifactMissing as err:
        pytest.skip(f"the pipeline has not run for this config hash: {err}")


# ---------------------------------------------------------------------------
# 1. Binding law 5
# ---------------------------------------------------------------------------


class TestTheUiCarriesNoComputedConstant:
    """The grep test of build spec 15, with its allowlist and a planted violation."""

    def test_the_check_is_clean_on_the_shipped_ui(self, repo_root):
        hits = dash_lint.check_ui_constants(repo_root)
        assert hits == [], "computed constants in the UI package:\n" + "\n".join(hits)

    def test_the_check_fires_on_a_planted_measurement(self, tmp_path):
        """A band level hard coded into a panel is exactly what this exists to catch."""
        package = tmp_path / dash_lint.UI_PACKAGE
        package.mkdir(parents=True)
        (package / dash_lint.UI_LAYOUT_MODULE).write_text("PANEL_HEIGHT_PX = 320\n", "utf-8")
        (package / "panel.py").write_text(
            "def band(sigma):\n    return 1.6449 * sigma\n", encoding="utf-8"
        )
        hits = dash_lint.check_ui_constants(tmp_path)
        assert len(hits) == 1
        assert "panel.py:2" in hits[0]
        assert "1.6449" in hits[0]

    def test_the_check_fires_on_a_presentation_name_that_is_not_one(self, tmp_path):
        """The allowlist is a suffix list, and it deliberately has no threshold in it."""
        package = tmp_path / dash_lint.UI_PACKAGE
        package.mkdir(parents=True)
        (package / dash_lint.UI_LAYOUT_MODULE).write_text(
            "PANEL_HEIGHT_PX = 320\nPEAK_LOAD_THRESHOLD = 33200.0\n", encoding="utf-8"
        )
        hits = dash_lint.check_ui_constants(tmp_path)
        assert len(hits) == 1
        assert "33200.0" in hits[0]

    def test_a_presentation_constant_outside_the_layout_module_is_still_a_violation(
        self, tmp_path
    ):
        """One module holds the presentation constants, so there is one place to read them."""
        package = tmp_path / dash_lint.UI_PACKAGE
        package.mkdir(parents=True)
        (package / dash_lint.UI_LAYOUT_MODULE).write_text("PANEL_HEIGHT_PX = 320\n", "utf-8")
        (package / "panel.py").write_text("OTHER_HEIGHT_PX = 240\n", encoding="utf-8")
        hits = dash_lint.check_ui_constants(tmp_path)
        assert len(hits) == 1
        assert "panel.py" in hits[0]

    def test_the_check_refuses_to_pass_when_there_is_nothing_to_check(self, tmp_path):
        """A law check that passes vacuously is worse than one that fails."""
        hits = dash_lint.check_ui_constants(tmp_path)
        assert len(hits) == 1
        assert "vacuously" in hits[0]

    def test_structural_literals_are_allowed_anywhere(self, tmp_path):
        package = tmp_path / dash_lint.UI_PACKAGE
        package.mkdir(parents=True)
        (package / dash_lint.UI_LAYOUT_MODULE).write_text("PANEL_HEIGHT_PX = 320\n", "utf-8")
        (package / "panel.py").write_text(
            "def first(rows):\n    return rows[0], rows[-1], len(rows) - 1\n", encoding="utf-8"
        )
        assert dash_lint.check_ui_constants(tmp_path) == []

    def test_the_documented_suffix_list_matches_the_enforced_one(self):
        """The layout module documents the allowlist; the linter enforces it. One list."""
        pytest.importorskip("matplotlib")
        from ufem.ui import layout

        assert layout.PRESENTATION_SUFFIXES == dash_lint.UI_PRESENTATION_SUFFIXES

    def test_the_lint_script_runs_the_ui_check(self, repo_root):
        """The check has to be in the script CI runs, not only in this file."""
        import inspect

        source = inspect.getsource(dash_lint.main)
        assert "check_ui_constants" in source


# ---------------------------------------------------------------------------
# 2. The graying rule
# ---------------------------------------------------------------------------


@pytest.mark.fullstack
class TestTheValidityGraying:
    """Where the panel grays out, and whether the warning names the corner it is in."""

    def test_the_design_median_is_inside_the_domain(self, store):
        from ufem.ui.predict import validity_verdict

        verdict = validity_verdict(store, store.design_midpoints)
        assert verdict.inside
        assert verdict.reason() == ""

    def test_the_verdict_agrees_with_the_project_wide_validity_check(self, store):
        """One answer to the domain question, not a second one written for the dashboard."""
        from ufem.config import FEATURE_ORDER
        from ufem.ui.predict import validity_verdict
        from ufem.validity import in_validity_domain

        bounds = store.design_bounds
        rng = np.random.default_rng(np.random.SeedSequence(20260830))
        rows = []
        verdicts = []
        for _ in range(32):
            values = {name: float(rng.uniform(*bounds[name])) for name in FEATURE_ORDER}
            rows.append([values[name] for name in FEATURE_ORDER])
            verdicts.append(validity_verdict(store, values).inside)
        expected = in_validity_domain(np.array(rows), store.repo_root, store.config)
        assert list(np.asarray(verdicts, dtype=bool)) == list(expected)

    def test_a_point_below_the_completion_threshold_is_grayed_and_says_why(self, store):
        from ufem.config import FEATURE_ORDER
        from ufem.ui.predict import validity_verdict

        bounds = store.design_bounds
        rng = np.random.default_rng(np.random.SeedSequence(4242))
        outside = None
        for _ in range(500):
            values = {name: float(rng.uniform(*bounds[name])) for name in FEATURE_ORDER}
            verdict = validity_verdict(store, values)
            if not verdict.inside:
                outside = verdict
                break
        assert outside is not None, (
            "no point of the executed design box fell outside the validity domain, which "
            "would mean the domain is the whole box and the graying rule is untestable."
        )
        assert outside.completion_probability < outside.threshold
        assert "below the stamped threshold" in outside.reason()

    def test_the_warning_names_the_censored_corner_from_the_audit_artifact(self, store):
        """The corner is read, not written: it is the enriched quantile bin of the campaign."""
        from ufem.ui.predict import validity_verdict

        corners = store.enriched_corners
        assert corners, "the censoring statistics record no significantly enriched input"
        corner = max(corners, key=lambda item: item["fail_rate"])
        values = dict(store.design_midpoints)
        values[corner["input"]] = 0.5 * (corner["low"] + corner["high"])
        verdict = validity_verdict(store, values)
        assert verdict.censored_corners
        reason = verdict.reason() if not verdict.inside else ""
        if not verdict.inside:
            assert "censored corner" in reason
            assert f"{corner['n_failed']} of {corner['n']}" in reason

    def test_a_point_outside_the_design_box_names_the_input_that_left_it(self, store):
        from ufem.ui.predict import validity_verdict

        values = dict(store.design_midpoints)
        low, high = store.design_bounds["c_nom_top_mm"]
        values["c_nom_top_mm"] = high + (high - low)
        verdict = validity_verdict(store, values)
        assert not verdict.inside
        assert not verdict.inside_design_box
        assert "outside the executed design range" in verdict.reason()

    def test_the_curve_figure_grays_the_line_and_says_it_did(self, store):
        """The figure is what the reader sees, so the graying is asserted on the figure."""
        from ufem.ui import figures, layout
        from ufem.ui.predict import predict

        current = predict(store, store.design_midpoints)
        grayed = figures.curve_figure(
            current.u_grid,
            current.force_mean,
            current.force_lower,
            current.force_upper,
            figures.FORCE_AXIS,
            "band",
            layout.CURVE_PANEL_HEIGHT_PX,
            grayed=True,
        )
        normal = figures.curve_figure(
            current.u_grid,
            current.force_mean,
            current.force_lower,
            current.force_upper,
            figures.FORCE_AXIS,
            "band",
            layout.CURVE_PANEL_HEIGHT_PX,
            grayed=False,
        )
        assert grayed.data[-1].line.color == layout.GRAYED_COLOR
        assert normal.data[-1].line.color == layout.PREDICTION_COLOR
        assert grayed.data[-1].line.color != normal.data[-1].line.color


# ---------------------------------------------------------------------------
# 3. The threshold recount
# ---------------------------------------------------------------------------


@pytest.mark.fullstack
class TestTheThresholdSlider:
    """What the reliability panel recomputes, against what the propagate stage recorded."""

    def test_the_configured_threshold_reproduces_the_recorded_numbers_exactly(self, store):
        from ufem.propagate import LIMIT_STATES, recompute_limit_state

        for state in LIMIT_STATES:
            recorded = store.subsample_limit_state(state.config_field)
            fresh = recompute_limit_state(
                store.mc_subsample, state.target, state.direction, recorded["threshold"]
            )
            for key, value in recorded.items():
                if isinstance(value, float) and math.isnan(value):
                    assert math.isnan(fresh[key]), (state.config_field, key)
                else:
                    assert fresh[key] == value, (state.config_field, key)

    def test_moving_the_threshold_moves_the_probability_monotonically(self, store):
        """A slider that did not order its answers would be reading the wrong column."""
        from ufem.propagate import LIMIT_STATES, recompute_limit_state, subsample_column

        state = LIMIT_STATES[0]
        column = store.mc_subsample[
            subsample_column(state.target, "mean")
        ].to_numpy(dtype=float)
        thresholds = np.linspace(column.min(), column.max(), 25)
        probabilities = [
            recompute_limit_state(
                store.mc_subsample, state.target, state.direction, float(threshold)
            )["pf_point"]
            for threshold in thresholds
        ]
        assert state.direction == "below"
        assert all(
            later >= earlier
            for earlier, later in zip(probabilities, probabilities[1:])
        )
        # The comparison is strict on both ends, which is what a limit state means: at the
        # sample minimum nothing has failed, and at the sample maximum every row but the one
        # sitting exactly on the threshold has.
        assert probabilities[0] == 0.0
        assert probabilities[-1] == (len(store.mc_subsample) - 1) / len(store.mc_subsample)

    def test_the_bound_is_never_below_the_point_estimate_at_any_threshold(self, store):
        from ufem.propagate import LIMIT_STATES, recompute_limit_state, subsample_column

        for state in LIMIT_STATES:
            column = store.mc_subsample[
                subsample_column(state.target, "mean")
            ].to_numpy(dtype=float)
            for threshold in np.linspace(column.min(), column.max(), 15):
                result = recompute_limit_state(
                    store.mc_subsample, state.target, state.direction, float(threshold)
                )
                assert result["pf_conservative"] >= result["pf_point"], (
                    state.config_field,
                    threshold,
                )

    def test_the_out_of_domain_fraction_does_not_depend_on_the_threshold(self, store):
        """It is a property of the sample, not of the limit state, and must read that way."""
        from ufem.propagate import LIMIT_STATES, recompute_limit_state, subsample_column

        state = LIMIT_STATES[0]
        column = store.mc_subsample[
            subsample_column(state.target, "mean")
        ].to_numpy(dtype=float)
        shares = {
            recompute_limit_state(
                store.mc_subsample, state.target, state.direction, float(threshold)
            )["out_of_domain_fraction"]
            for threshold in np.linspace(column.min(), column.max(), 5)
        }
        assert len(shares) == 1
        assert shares.pop() == store.propagation["subsample"]["out_of_domain_fraction"]


# ---------------------------------------------------------------------------
# 4. The export
# ---------------------------------------------------------------------------


@pytest.mark.fullstack
class TestTheExport:
    """What the export button writes, read back."""

    @staticmethod
    def _round_trip(store):
        from ufem.ui.predict import export_payload, predict

        prediction = predict(store, store.design_midpoints)
        payload = export_payload(store, prediction)
        return prediction, json.loads(json.dumps(payload, sort_keys=True))

    def test_the_payload_round_trips_through_json(self, store):
        _prediction, restored = self._round_trip(store)
        assert restored["kind"] == "ufem_lab_prediction"
        assert set(restored["inputs"]) == set(store.design_midpoints)

    def test_the_curves_survive_the_round_trip_to_the_last_bit(self, store):
        prediction, restored = self._round_trip(store)
        for key, values in (
            ("u_mm", prediction.u_grid),
            ("force_N", prediction.force_mean),
            ("force_lower_N", prediction.force_lower),
            ("force_upper_N", prediction.force_upper),
            ("damage", prediction.damage_mean),
        ):
            assert np.array_equal(np.asarray(restored["curves"][key]), values), key

    def test_the_payload_carries_the_config_hash_and_every_stage_manifest(self, store):
        from ufem.ui.store import REQUIRED_STAGES

        _prediction, restored = self._round_trip(store)
        assert restored["config_sha256"] == store.config_sha256
        assert set(restored["manifests"]) == {stage for stage, _how in REQUIRED_STAGES}
        for stage, record in restored["manifests"].items():
            assert record["outputs"], stage
            assert all(len(digest) == 64 for digest in record["outputs"].values()), stage

    def test_every_manifest_digest_in_the_export_resolves_to_a_file_on_disk(self, store):
        """Binding law 5: the hashes in an exported prediction have to be real."""
        from ufem.manifest import sha256_file

        _prediction, restored = self._round_trip(store)
        for stage, record in restored["manifests"].items():
            directory = store.repo_root / record["directory"]
            for name, digest in record["outputs"].items():
                path = directory / name
                assert path.is_file(), f"{stage}: {path}"
                assert sha256_file(path) == digest, f"{stage}: {name}"

    def test_the_scalar_intervals_bracket_their_own_means(self, store):
        prediction, restored = self._round_trip(store)
        for readout in prediction.scalars:
            record = restored["scalars"][readout.name]
            assert record["lower"] <= record["mean"] <= record["upper"], readout.name
            assert record["sigma"] > 0.0, readout.name

    def test_the_export_states_the_band_level_it_drew(self, store):
        _prediction, restored = self._round_trip(store)
        assert restored["band_alpha"] == store.band_alpha
        assert "jackknife+" in restored["band_construction"]


# ---------------------------------------------------------------------------
# 5. The latency budget, server side
# ---------------------------------------------------------------------------


#: Build spec 15's budget for the slider to repaint. This half of it, the server side, is
#: prediction plus figure construction plus JSON serialization, which is the work a slider move
#: costs before a byte leaves the process.
LATENCY_BUDGET_MS = 50.0

#: Positions to measure over, and how many to discard first. The discards are warm up: the
#: first call through a fresh GPyTorch module is an order of magnitude slower than the rest,
#: and a dashboard pays that once at startup rather than on every interaction.
LATENCY_POSITIONS = 100
LATENCY_WARMUP = 10


def _slider_positions(store, n_positions: int, entropy: int) -> list[dict[str, float]]:
    """Seeded positions inside the executed design box, so the measurement is repeatable."""
    from ufem.config import FEATURE_ORDER

    bounds = store.design_bounds
    generator = np.random.default_rng(np.random.SeedSequence(entropy))
    return [
        {name: float(generator.uniform(*bounds[name])) for name in FEATURE_ORDER}
        for _ in range(n_positions)
    ]


def _serve_once(store, values: dict[str, float]) -> tuple[float, int]:
    """One slider move, server side: predict, build both figures, serialize. Milliseconds."""
    import plotly.utils

    from ufem.ui import figures, layout
    from ufem.ui.predict import predict

    started = time.perf_counter()
    current = predict(store, values)
    grayed = not current.validity.inside
    force = figures.curve_figure(
        current.u_grid,
        current.force_mean,
        current.force_lower,
        current.force_upper,
        figures.FORCE_AXIS,
        "band",
        layout.CURVE_PANEL_HEIGHT_PX,
        grayed=grayed,
    )
    damage = figures.curve_figure(
        current.u_grid,
        current.damage_mean,
        current.damage_lower,
        current.damage_upper,
        figures.DAMAGE_AXIS,
        "band",
        layout.DAMAGE_PANEL_HEIGHT_PX,
        grayed=grayed,
    )
    blob = json.dumps(
        [force.to_plotly_json(), damage.to_plotly_json()], cls=plotly.utils.PlotlyJSONEncoder
    )
    return (time.perf_counter() - started) * 1000.0, len(blob)


@pytest.mark.fullstack
class TestTheServerSideLatencyBudget:
    """Build spec 15: under 50 ms from a slider move to a serialized repaint payload."""

    def test_the_median_slider_move_is_inside_the_budget(self, store, record_property):
        positions = _slider_positions(store, LATENCY_POSITIONS + LATENCY_WARMUP, 20260830)
        timings = [_serve_once(store, values)[0] for values in positions]
        measured = np.array(timings[LATENCY_WARMUP:])
        median = float(np.median(measured))
        record_property("median_ms", median)
        record_property("p90_ms", float(np.percentile(measured, 90)))
        assert measured.size == LATENCY_POSITIONS
        assert median < LATENCY_BUDGET_MS, (
            f"median server side slider latency {median:.1f} ms over {measured.size} seeded "
            f"positions, above the {LATENCY_BUDGET_MS:.0f} ms budget of build spec 15. The "
            "measured value is recorded in docs/ENGINEERING_LOG.md; a regression here means "
            "the panel started recomputing something it used to have."
        )

    def test_the_repaint_payload_is_small_enough_to_be_pushed(self, store):
        """A budget met by a payload nobody can send is not a budget met."""
        _elapsed, size = _serve_once(store, store.design_midpoints)
        assert size < 1 << 20


# ---------------------------------------------------------------------------
# The browser half, marked slow
# ---------------------------------------------------------------------------


#: What the playwright test asserts on this machine. It is generous against the server side
#: budget on purpose: it includes the websocket round trip, Plotly's own repaint, and whatever
#: the machine was doing at the time. The measured value is recorded in
#: docs/ENGINEERING_LOG.md, and this is the bound a regression would have to cross to be a
#: regression rather than noise.
BROWSER_LATENCY_BUDGET_MS = 500.0

#: What the browser is asked for: the drawn path of the last line in the first plot, which is
#: the predicted mean curve. The rendered path rather than the underlying data, deliberately.
#: Plotly 3 may hand a trace its ``y`` as a typed array specification object rather than as a
#: JavaScript array, so reading the data proves nothing about what is on the screen, while the
#: SVG path is the repaint itself. A path that changed is a repaint that happened.
#: Parenthesized so it can be both evaluated as a function and called inline inside a larger
#: predicate expression, which is how ``wait_for_function`` takes it.
MEAN_CURVE_PATH_JS = """(() => {
  const plot = document.querySelectorAll('.js-plotly-plot')[0];
  if (!plot) { return ''; }
  const lines = plot.querySelectorAll('.scatterlayer .js-line');
  return lines.length ? lines[lines.length - 1].getAttribute('d') : '';
})"""

#: Generous, because a headless browser launching on a machine that is also running the server
#: is not a quiet environment.
BROWSER_READY_TIMEOUT_MS = 90000


@pytest.mark.slow
@pytest.mark.fullstack
def test_the_dashboard_repaints_in_a_real_browser(repo_root, record_property):
    """Launch `ufem lab` headless, move a slider, assert the plot repainted, and time it."""
    pytest.importorskip("playwright")
    from capture_ui_gif import (
        VIEWPORT_HEIGHT_PX,
        VIEWPORT_WIDTH_PX,
        LabServer,
        chromium_available,
    )

    if not chromium_available():
        pytest.skip(
            "the playwright chromium build is not installed; run "
            "`python -m playwright install chromium` to enable this test"
        )
    from playwright.sync_api import sync_playwright

    with LabServer(repo_root) as server:
        if server.skip_reason is not None:
            pytest.skip(server.skip_reason)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page(
                    viewport={"width": VIEWPORT_WIDTH_PX, "height": VIEWPORT_HEIGHT_PX}
                )
                page.goto(server.url, wait_until="networkidle")
                page.wait_for_selector("text=Predict", timeout=BROWSER_READY_TIMEOUT_MS)
                page.wait_for_selector(
                    ".scatterlayer .js-line", timeout=BROWSER_READY_TIMEOUT_MS
                )
                page.wait_for_function(
                    f"{MEAN_CURVE_PATH_JS}().length > 0", timeout=BROWSER_READY_TIMEOUT_MS
                )
                before = page.evaluate(MEAN_CURVE_PATH_JS)
                assert before, "the predicted mean curve was never drawn"

                slider = page.locator(".q-slider").first
                box = slider.bounding_box()
                assert box is not None, "the strength slider has no box to click in"
                started = time.perf_counter()
                page.mouse.click(box["x"] + box["width"] * 0.85, box["y"] + box["height"] / 2)
                page.wait_for_function(
                    f"previous => {MEAN_CURVE_PATH_JS}() !== previous",
                    arg=before,
                    timeout=BROWSER_READY_TIMEOUT_MS,
                )
                elapsed = (time.perf_counter() - started) * 1000.0
                after = page.evaluate(MEAN_CURVE_PATH_JS)
            finally:
                browser.close()

    record_property("browser_latency_ms", elapsed)
    print(f"\nbrowser slider to repaint: {elapsed:.0f} ms")
    assert after != before, "the curve did not change when the slider moved"
    assert elapsed < BROWSER_LATENCY_BUDGET_MS, (
        f"slider to repaint took {elapsed:.0f} ms in a real browser, above the "
        f"{BROWSER_LATENCY_BUDGET_MS:.0f} ms bound. The server side half of the budget is "
        "asserted separately at 50 ms; this bound covers the websocket round trip and the "
        "Plotly repaint on top of it."
    )
