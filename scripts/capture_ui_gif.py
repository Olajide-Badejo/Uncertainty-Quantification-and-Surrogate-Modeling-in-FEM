"""Record the README GIF of build spec section 15.1, headless, from the real dashboard.

Launches ``ufem lab`` on a scratch port, drives a scripted interaction with playwright,
captures one screenshot per step, and assembles ``docs/media/ufem_lab.gif`` with Pillow.

The recording is a demonstration, not a screensaver. It answers two questions a reader has
before they will install anything: what is this for, and how do I drive it. So the interaction
is a scripted story with five beats, each held long enough to read:

1. **Predict.** The strength slider sweeps its design range low to high and the calibrated
   curve and its simultaneous band morph under it; then it settles at a middle value and the
   page scrolls to the quantities of interest so their jackknife plus intervals are readable.
2. **The censored corner.** The inputs are driven into the corner where the campaign failed.
   The validity warning appears, naming the completion probability and the corner, and every
   curve grays out. Then back to a design point the model is allowed to speak about.
3. **Dataset.** A completed run is clicked in the design matrix, the selection marker moves to
   it, and the page scrolls to the finite element curve drawn against the surrogate's
   prediction at the same three inputs. This is the view that convinces.
4. **Reliability.** The limit state threshold slider sweeps and the failure probability, its
   standard error, its Wilson interval and its conservative bound recount live.
5. **Model card.** A closing beat on the provenance the whole thing rests on.

Nothing on screen is staged: every panel is the dashboard reading the artifact store, and the
only thing this script adds is the order the panels are visited in and how long each is held.

Framing. The browser viewport is :data:`VIEWPORT_WIDTH_PX` by :data:`VIEWPORT_HEIGHT_PX` and
is downscaled to :data:`TARGET_WIDTH_PX`, which fixes the defect the first capture shipped
with: it was recorded in a viewport shorter than the panels and clipped them. Every beat now
names the scroll offset that puts its subject wholly inside the frame, and those offsets were
measured against the real layout rather than guessed. ``--frames DIR`` writes the captured
frames out as PNGs so that claim can be checked by looking, which is the only way a framing
claim can be checked.

Two deviations from build spec 15.1, both deliberate and both recorded in
docs/DESIGN_DECISIONS.md:

1. **Pillow rather than ffmpeg.** ffmpeg is not installed on this machine and is not a Python
   dependency this project is willing to acquire for one figure. Pillow is already in the
   stack. What is lost is ffmpeg's two pass palette generation; what replaces it is a single
   global adaptive palette quantized from a strided sample of the frames, which is the same
   idea with a cheaper sampler, plus Pillow's own interframe bounding box optimization. The
   measured result is in docs/ENGINEERING_LOG.md.
2. **Step driven rather than real time.** A playwright screenshot costs far more than a frame
   interval, so capturing at wall clock 12 fps would either drop frames or slow the
   interaction to a crawl. The interaction is scripted as a sequence of steps instead, one
   frame per step, played back at :data:`FRAME_RATE` frames per second. The GIF's duration is
   therefore frames divided by frame rate exactly, and it is asserted rather than hoped for.

Run it directly. Exit 0 is clean, exit 1 names the failure. Nothing here falls back: if the
browser is missing, the script says so and stops rather than writing a smaller GIF from
whatever it managed to capture, and if a scripted interaction does not take effect, it raises
rather than recording a beat that shows nothing happening.
"""

from __future__ import annotations

import io
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageSequence

from check_file_sizes import LIMIT_BYTES as TRACKED_FILE_LIMIT_BYTES

#: Build spec 15.1. Frame rate, output width, and the ceiling the file must come in under.
FRAME_RATE = 12
TARGET_WIDTH_PX = 960
SPEC_SIZE_LIMIT_BYTES = 15 * 1024 * 1024

#: The limit that actually binds. Build spec 15.1 allows 15 MB, and build spec 3.3 allows no
#: tracked file over 5 MB with no exemption for this one, so the smaller of the two is the
#: gate. Checking the spec's number alone would have let a capture through that the file size
#: gate then rejected, which is a failure discovered one commit too late.
SIZE_LIMIT_BYTES = min(SPEC_SIZE_LIMIT_BYTES, TRACKED_FILE_LIMIT_BYTES)

#: Build spec 15.1 asks for 12 to 20 seconds, and the story below uses most of that window.
MIN_DURATION_S = 12.0
MAX_DURATION_S = 20.0

#: The browser window the interaction is driven in, then downscaled to
#: :data:`TARGET_WIDTH_PX`. The height is the measurement that matters: the tallest thing any
#: beat has to show whole is the Predict panel's quantity of interest table at 457 px under a
#: 260 px damage panel, and the Dataset panel's two 380 px overlays with their caption. Both
#: fit inside this window, which the previous 900 px window did not, and ``--frames`` is how
#: that is verified rather than asserted.
VIEWPORT_WIDTH_PX = 1280
VIEWPORT_HEIGHT_PX = 960

#: Colors in the shared palette. 128 is the point on this content where a further halving
#: starts to band the plot fills; the measurement is in docs/ENGINEERING_LOG.md.
PALETTE_COLORS = 128

#: How many frames are sampled to build that palette. Every frame would be exact and slow.
PALETTE_SAMPLE_FRAMES = 24

#: Milliseconds to let the dashboard settle after an interaction before the shutter. The
#: server side budget is 50 ms and the websocket round trip is the rest.
SETTLE_MS = 90
SCROLL_SETTLE_MS = 260
TAB_SETTLE_MS = 1100

#: How long to wait for the server to answer, and for the first plot to exist.
SERVER_TIMEOUT_S = 180.0
SELECTOR_TIMEOUT_MS = 90000

OUTPUT = Path("docs/media/ufem_lab.gif")


class CaptureFailed(RuntimeError):
    """The capture could not be completed, and no partial GIF will be written."""


def chromium_available() -> bool:
    """True when playwright's chromium build is installed and launchable.

    A capability probe, not a fallback. It answers one question, whether the browser exists,
    and every caller then decides for itself: the test skips with a named reason, the capture
    script exits nonzero. Neither of them proceeds without a browser.
    """
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as playwright:
            path = playwright.chromium.executable_path
    except (PlaywrightError, OSError):
        return False
    return bool(path) and Path(path).exists()


def free_port() -> int:
    """An unused local port, so a capture never collides with a dashboard someone is using."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class LabServer:
    """`ufem lab` as a context manager: start it, wait for it, stop it, report its output.

    Used by the capture script and by the playwright test in ``tests/test_ui.py``, so there is
    one way to start the dashboard for automation and one place where its startup is waited
    for. ``skip_reason`` is set instead of raising when the pipeline has not run, because the
    test skips on that and the script exits nonzero on it, and those are two different
    responses to one condition.
    """

    repo_root: Path
    port: int = field(default_factory=free_port)
    process: subprocess.Popen | None = field(default=None, repr=False)
    output: str = ""
    skip_reason: str | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def __enter__(self) -> LabServer:
        executable = self.repo_root / ".venv" / "Scripts" / "python.exe"
        interpreter = str(executable) if executable.is_file() else sys.executable
        # NiceGUI switches into its own screen test mode when PYTEST_CURRENT_TEST is set, and
        # then demands a port it expects that harness to have provided. The dashboard being
        # launched here is a separate process serving the real app, so the marker is dropped
        # from its environment: it is not running under pytest, whatever launched it is.
        environment = {
            name: value
            for name, value in os.environ.items()
            if name not in ("PYTEST_CURRENT_TEST", "PYTEST_VERSION")
        }
        self.process = subprocess.Popen(
            [interpreter, "-m", "ufem.runner", "lab", "--no-browser", "--port", str(self.port)],
            cwd=str(self.repo_root),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.time() + SERVER_TIMEOUT_S
        while time.time() < deadline:
            if self.process.poll() is not None:
                self.output = self.process.stdout.read() if self.process.stdout else ""
                if "unavailable" in self.output or "LabArtifactMissing" in self.output:
                    self.skip_reason = (
                        f"`ufem lab` could not start against this artifact store: {self.output}"
                    )
                    return self
                raise CaptureFailed(
                    f"`ufem lab` exited with {self.process.returncode} before serving:\n"
                    f"{self.output}"
                )
            try:
                with urllib.request.urlopen(self.url, timeout=2.0) as response:
                    if response.status == 200:
                        return self
            except (urllib.error.URLError, OSError, TimeoutError):
                time.sleep(1.0)
        raise CaptureFailed(
            f"`ufem lab` did not answer on {self.url} within {SERVER_TIMEOUT_S:.0f} s."
        )

    def __exit__(self, *_exc: object) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.output = self.process.communicate(timeout=30)[0] or self.output
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.output = self.process.communicate()[0] or self.output


# ---------------------------------------------------------------------------
# Where each beat is framed
# ---------------------------------------------------------------------------

#: Document scroll offset, in CSS pixels, that puts each beat's subject wholly in the window.
#: Measured against the rendered layout, not guessed: the Predict panel's table starts at 1058
#: and is 457 tall, the Dataset overlays start at 1553 and end at 2329 with the run picker
#: under them, and the Reliability slider, its recount table and its density panel span 678 to
#: 1478. Every one of those spans is shorter than :data:`VIEWPORT_HEIGHT_PX`.
SCROLL_TOP = 0
SCROLL_PREDICT_SCALARS = 640
SCROLL_DATASET_OVERLAY = 1440
SCROLL_RELIABILITY_RECOUNT = 620

#: The frame budget, beat by beat. Written as constants rather than as literals in the loop so
#: the duration assertion at the end is a statement about this list. At 12 fps nominal, and a
#: GIF delay quantized to hundredths of a second, these play back at 80 ms each.
OPEN_FRAMES = 12
SWEEP_FRAMES = 30
SWEEP_HOLD_FRAMES = 5
MID_HOLD_FRAMES = 6
SCROLL_FRAMES = 3
SCALARS_HOLD_FRAMES = 16
RETURN_FRAMES = 2
CORNER_FRAMES = 12
CORNER_HOLD_FRAMES = 15
RECOVER_FRAMES = 7
RECOVER_HOLD_FRAMES = 5
DATASET_FRAMES = 14
SELECT_HOLD_FRAMES = 9
OVERLAY_SCROLL_FRAMES = 4
OVERLAY_HOLD_FRAMES = 18
RELIABILITY_FRAMES = 10
RECOUNT_SCROLL_FRAMES = 2
RECOUNT_HOLD_FRAMES = 5
THRESHOLD_FRAMES = 20
THRESHOLD_HOLD_FRAMES = 8
CARD_FRAMES = 14

#: Splom cell used to place the click of beat 3: the first input on x, the second on y, which
#: is the top left panel of the design matrix and the one with the most separation between the
#: completed and the failed clouds.
CLICK_X_AXIS = "xaxis"
CLICK_Y_AXIS = "yaxis2"

#: Pixel coordinates of one completed run in the design matrix, computed from Plotly's own
#: axis mapping rather than from a guessed offset into the panel. The point chosen is the
#: completed run closest to the centre of the executed design, so the beat shows a typical
#: member rather than whichever run happens to sit at an edge.
CLICK_POINT_JS = """([xName, yName]) => {
  const gd = document.querySelectorAll('.js-plotly-plot')[0];
  if (!gd || !gd._fullData || !gd._fullData.length) { return null; }
  const trace = gd._fullData[0];
  const dims = trace.dimensions;
  if (!dims || dims.length < 2) { return null; }
  const xs = Array.from(dims[0].values);
  const ys = Array.from(dims[1].values);
  const mean = v => v.reduce((a, b) => a + b, 0) / v.length;
  const sd = (v, m) => Math.sqrt(v.reduce((a, b) => a + (b - m) * (b - m), 0) / v.length);
  const mx = mean(xs), my = mean(ys);
  const sx = sd(xs, mx) || 1, sy = sd(ys, my) || 1;
  let best = 0, bestD = Infinity;
  for (let i = 0; i < xs.length; i += 1) {
    const dx = (xs[i] - mx) / sx, dy = (ys[i] - my) / sy;
    const d = dx * dx + dy * dy;
    if (d < bestD) { bestD = d; best = i; }
  }
  const xa = gd._fullLayout[xName], ya = gd._fullLayout[yName];
  if (!xa || !ya || typeof xa.l2p !== 'function') { return null; }
  const rect = gd.getBoundingClientRect();
  return {
    x: rect.left + xa._offset + xa.l2p(xs[best]),
    y: rect.top + ya._offset + ya.l2p(ys[best]),
    n: xs.length
  };
}"""


def _shot(page, frames: list[Image.Image], count: int = 1) -> None:
    """One screenshot, repeated ``count`` times in the frame list."""
    image = Image.open(io.BytesIO(page.screenshot(type="png"))).convert("RGB")
    height = round(image.height * TARGET_WIDTH_PX / image.width)
    resized = image.resize((TARGET_WIDTH_PX, height), Image.LANCZOS)
    frames.extend(resized for _ in range(count))


def _scroll(page, offset: int) -> None:
    """Put the document at an absolute offset, which is how a beat is framed."""
    page.evaluate("offset => window.scrollTo(0, offset)", offset)
    page.wait_for_timeout(SCROLL_SETTLE_MS)


def _scroll_over(page, frames: list[Image.Image], start: int, end: int, steps: int) -> None:
    """Scroll from one framing to another over ``steps`` frames, so the move is legible."""
    for step in range(1, steps + 1):
        _scroll(page, round(start + (end - start) * step / steps))
        _shot(page, frames)


def _set_slider(page, index: int, fraction: float) -> None:
    """Click one slider at a fraction of its track, which is how a value is set here."""
    slider = page.locator(".q-slider").nth(index)
    box = slider.bounding_box()
    if box is None:
        raise CaptureFailed(f"slider {index} has no bounding box to click in")
    page.mouse.click(box["x"] + box["width"] * fraction, box["y"] + box["height"] / 2)
    page.wait_for_timeout(SETTLE_MS)


def _sweep(page, frames: list[Image.Image], index: int, low: float, high: float, steps: int):
    """Move one slider across a span, one frame per step."""
    for step in range(steps):
        _set_slider(page, index, low + (high - low) * step / max(steps - 1, 1))
        _shot(page, frames)


def _warning_text(page) -> str:
    """The validity warning as it currently reads, empty when the point is inside."""
    found = page.locator("text=Outside the validity domain")
    return found.first.inner_text() if found.count() else ""


def _caption_text(page) -> str:
    """The dataset panel's selected run caption, which is what a click has to change."""
    found = page.locator("text=Completion probability")
    return found.last.inner_text() if found.count() else ""


def capture_frames(page, frames_dir: Path | None = None) -> list[Image.Image]:
    """Drive the scripted interaction and return the frames, in order."""
    frames: list[Image.Image] = []

    # -- beat 1: the calibrated prediction, and the slider that drives it ----
    page.wait_for_selector("text=Predict", timeout=SELECTOR_TIMEOUT_MS)
    page.wait_for_selector(".js-plotly-plot", timeout=SELECTOR_TIMEOUT_MS)
    page.wait_for_timeout(TAB_SETTLE_MS)
    _scroll(page, SCROLL_TOP)
    _shot(page, frames, OPEN_FRAMES)

    _sweep(page, frames, 0, 0.04, 0.96, SWEEP_FRAMES)
    _shot(page, frames, SWEEP_HOLD_FRAMES)
    _set_slider(page, 0, 0.5)
    _shot(page, frames, MID_HOLD_FRAMES)

    # The quantities of interest with their jackknife plus intervals, which are the reason to
    # trust the picture above them and are below the fold at every framing that shows it.
    _scroll_over(page, frames, SCROLL_TOP, SCROLL_PREDICT_SCALARS, SCROLL_FRAMES)
    _shot(page, frames, SCALARS_HOLD_FRAMES)
    _scroll_over(page, frames, SCROLL_PREDICT_SCALARS, SCROLL_TOP, RETURN_FRAMES)

    # -- beat 2: the censored corner ----------------------------------------
    # High strength and low top cover is where the campaign died: the audit measured 63 of 100
    # designed runs failing in the top strength quartile and 76 of 100 in the lowest top cover
    # quartile. Driving there is the fastest way to show that the model refuses to speak.
    steps = CORNER_FRAMES // 2
    _sweep(page, frames, 0, 0.62, 0.97, steps)
    _sweep(page, frames, 2, 0.42, 0.02, CORNER_FRAMES - steps)
    warning = _warning_text(page)
    if not warning:
        raise CaptureFailed(
            "the validity warning did not appear at the corner this beat drives into, so the "
            "recording would show the graying without the reason for it. Either the domain "
            "moved or the slider indices did; check src/ufem/ui/app.py before recapturing."
        )
    print(f"capture_ui_gif: validity warning reads {warning.splitlines()[0]!r}")
    _shot(page, frames, CORNER_HOLD_FRAMES)

    recover = RECOVER_FRAMES // 2
    _sweep(page, frames, 2, 0.10, 0.55, recover)
    _sweep(page, frames, 0, 0.90, 0.45, RECOVER_FRAMES - recover)
    if _warning_text(page):
        raise CaptureFailed(
            "the validity warning is still showing after the recovery sweep, so the beat ends "
            "on a grayed prediction. Check the domain before recapturing."
        )
    _shot(page, frames, RECOVER_HOLD_FRAMES)

    # -- beat 3: a finite element run against the surrogate ------------------
    page.click("text=Dataset")
    page.wait_for_timeout(TAB_SETTLE_MS)
    page.wait_for_selector(".js-plotly-plot", timeout=SELECTOR_TIMEOUT_MS)
    _scroll(page, SCROLL_TOP)
    _shot(page, frames, DATASET_FRAMES)

    before = _caption_text(page)
    target = page.evaluate(CLICK_POINT_JS, [CLICK_X_AXIS, CLICK_Y_AXIS])
    if target is None:
        raise CaptureFailed(
            "the design matrix did not expose a completed point to click. The panel is a "
            "Plotly splom and this script reads its axis mapping to place the click; if the "
            "figure changed, update CLICK_POINT_JS rather than clicking a guessed pixel."
        )
    print(f"capture_ui_gif: clicking a completed run of {int(target['n'])} in the design matrix")
    page.mouse.click(float(target["x"]), float(target["y"]))
    page.wait_for_timeout(TAB_SETTLE_MS)
    after = _caption_text(page)
    if not after or after == before:
        raise CaptureFailed(
            "clicking the design matrix did not change the selected run, so the beat would "
            f"show a click with no effect. The caption still reads {before!r}."
        )
    _shot(page, frames, SELECT_HOLD_FRAMES)

    _scroll_over(page, frames, SCROLL_TOP, SCROLL_DATASET_OVERLAY, OVERLAY_SCROLL_FRAMES)
    _shot(page, frames, OVERLAY_HOLD_FRAMES)

    # -- beat 4: the limit state threshold, recounting ------------------------
    _scroll(page, SCROLL_TOP)
    page.click("text=Reliability")
    page.wait_for_timeout(TAB_SETTLE_MS)
    page.wait_for_selector(".js-plotly-plot", timeout=SELECTOR_TIMEOUT_MS)
    _shot(page, frames, RELIABILITY_FRAMES)
    _scroll_over(
        page, frames, SCROLL_TOP, SCROLL_RELIABILITY_RECOUNT, RECOUNT_SCROLL_FRAMES
    )
    _shot(page, frames, RECOUNT_HOLD_FRAMES)
    _sweep(page, frames, 0, 0.18, 0.86, THRESHOLD_FRAMES)
    _shot(page, frames, THRESHOLD_HOLD_FRAMES)

    # -- beat 5: the provenance it all rests on -------------------------------
    _scroll(page, SCROLL_TOP)
    page.click("text=Model card")
    page.wait_for_timeout(TAB_SETTLE_MS)
    _shot(page, frames, CARD_FRAMES)

    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)
        for existing in frames_dir.glob("frame_*.png"):
            existing.unlink()
        seen: list[int] = []
        previous: Image.Image | None = None
        for position, frame in enumerate(frames):
            if previous is not None and frame is previous:
                continue
            previous = frame
            seen.append(position)
            frame.save(frames_dir / f"frame_{position:04d}.png")
        print(f"capture_ui_gif: wrote {len(seen)} distinct frames to {frames_dir}")
    return frames


def playback(path: Path) -> tuple[int, float]:
    """The stored frame count and playback seconds of a written GIF, read back from it.

    Both differ from what was asked for, and the difference is not an error. Pillow merges a
    run of identical frames into one and accumulates their delays, so a hold of fourteen
    frames is stored once; and the GIF format carries a delay in hundredths of a second, so a
    12 fps interval of 83.3 ms is written as 80 ms and the file plays slightly faster than the
    nominal rate. The duration this returns is what a viewer will see, which is the number the
    window of build spec 15.1 is a statement about.
    """
    with Image.open(path) as image:
        total = 0
        stored = 0
        for frame in ImageSequence.Iterator(image):
            total += int(frame.info.get("duration", 0))
            stored += 1
    return stored, total / 1000.0


def assemble_gif(frames: list[Image.Image], path: Path) -> int:
    """Quantize onto one shared palette and write the GIF. Returns its size in bytes.

    One palette for every frame is what lets Pillow write later frames as a bounding box of
    what changed rather than as a full image, which is where almost all of the compression on
    a dashboard recording comes from: most of the screen is identical from frame to frame.
    """
    if not frames:
        raise CaptureFailed("no frames were captured, so there is nothing to assemble.")
    stride = max(len(frames) // PALETTE_SAMPLE_FRAMES, 1)
    sample = frames[::stride]
    strip = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for position, frame in enumerate(sample):
        strip.paste(frame, (0, position * frame.height))
    palette = strip.quantize(colors=PALETTE_COLORS, method=Image.Quantize.MEDIANCUT)
    quantized = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=quantized[1:],
        duration=round(1000 / FRAME_RATE),
        loop=0,
        optimize=True,
        disposal=1,
    )
    return path.stat().st_size


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    frames_dir: Path | None = None
    if "--frames" in argv:
        position = argv.index("--frames")
        if position + 1 >= len(argv):
            print("capture_ui_gif: --frames needs a directory", file=sys.stderr)
            return 1
        frames_dir = Path(argv[position + 1]).resolve()
        del argv[position : position + 2]
    root = Path(argv[0]).resolve() if argv else Path(__file__).resolve().parents[1]
    if not chromium_available():
        print(
            "capture_ui_gif: playwright's chromium build is not installed. Run "
            "`python -m playwright install chromium`. The GIF is a committed deliverable of "
            "build spec 15.1 and this script will not write a substitute for it.",
            file=sys.stderr,
        )
        return 1
    from playwright.sync_api import sync_playwright

    started = time.perf_counter()
    with LabServer(root) as server:
        if server.skip_reason is not None:
            print(f"capture_ui_gif: {server.skip_reason}", file=sys.stderr)
            return 1
        print(f"capture_ui_gif: serving {server.url}")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page(
                    viewport={"width": VIEWPORT_WIDTH_PX, "height": VIEWPORT_HEIGHT_PX},
                    device_scale_factor=1,
                )
                page.goto(server.url, wait_until="networkidle")
                frames = capture_frames(page, frames_dir)
            finally:
                browser.close()
    capture_seconds = time.perf_counter() - started

    target = root / OUTPUT
    previous = target.stat().st_size if target.is_file() else None
    scratch = target.with_name(target.name + ".part")
    size = assemble_gif(frames, scratch)
    stored, duration = playback(scratch)
    print(
        f"capture_ui_gif: {len(frames)} captured frames stored as {stored}, {duration:.2f} s "
        f"of playback at a nominal {FRAME_RATE} fps, {frames[0].width} by "
        f"{frames[0].height} px, {size / 1024 / 1024:.2f} MB, captured in "
        f"{capture_seconds:.1f} s"
        + (f" (previous {previous / 1024 / 1024:.2f} MB)" if previous else "")
    )
    failures = []
    if not MIN_DURATION_S <= duration <= MAX_DURATION_S:
        failures.append(
            f"duration {duration:.2f} s is outside the {MIN_DURATION_S:.0f} to "
            f"{MAX_DURATION_S:.0f} second window of build spec 15.1"
        )
    if size > SIZE_LIMIT_BYTES:
        failures.append(
            f"{size / 1024 / 1024:.2f} MB exceeds the {SIZE_LIMIT_BYTES / 1024 / 1024:.0f} MB "
            "ceiling that binds here, which is the smaller of build spec 15.1 and the tracked "
            "file limit of build spec 3.3; drop the frame count, the palette or the width and "
            "record the change in docs/DESIGN_DECISIONS.md"
        )
    if failures:
        scratch.unlink(missing_ok=True)
        for failure in failures:
            print(f"capture_ui_gif: {failure}", file=sys.stderr)
        return 1
    shutil.move(str(scratch), str(target))
    print(f"capture_ui_gif: wrote {OUTPUT.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
