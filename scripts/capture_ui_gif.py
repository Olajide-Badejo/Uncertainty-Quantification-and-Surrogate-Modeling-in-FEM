"""Record the README GIF of build spec section 15.1, headless, from the real dashboard.

Launches ``ufem lab`` on a scratch port, drives a scripted interaction with playwright,
captures one screenshot per step, and assembles ``docs/media/ufem_lab.gif`` with Pillow.

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
whatever it managed to capture.
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

#: Build spec 15.1. Frame rate, width, and the ceiling the file must come in under.
FRAME_RATE = 12
TARGET_WIDTH_PX = 960
SIZE_LIMIT_BYTES = 15 * 1024 * 1024

#: Build spec 15.1 asks for 12 to 20 seconds. At :data:`FRAME_RATE` this is the frame count
#: that puts the result in the middle of that window.
MIN_DURATION_S = 12.0
MAX_DURATION_S = 20.0

#: The browser window the interaction is driven in. Wider than the output so the dashboard
#: lays out as it does on a real screen and is then downscaled, rather than being rendered
#: into a narrow viewport that reflows the panels.
VIEWPORT_WIDTH_PX = 1440
VIEWPORT_HEIGHT_PX = 900

#: Colors in the shared palette. 128 is the point on this content where a further halving
#: starts to band the plot fills; the measurement is in docs/ENGINEERING_LOG.md.
PALETTE_COLORS = 128

#: How many frames are sampled to build that palette. Every frame would be exact and slow.
PALETTE_SAMPLE_FRAMES = 24

#: Milliseconds to let the dashboard settle after an interaction before the shutter. The
#: server side budget is 50 ms and the websocket round trip is the rest.
SETTLE_MS = 90
TAB_SETTLE_MS = 900

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


#: The scripted interaction of build spec 15.1, as (what to do, how many frames to hold).
#: Written as data so the frame budget is legible and so the duration assertion below is a
#: statement about this list rather than about whatever the loop happened to do.
INTRO_FRAMES = 14
SWEEP_FRAMES = 48
SWEEP_HOLD_FRAMES = 10
DATASET_FRAMES = 14
SCROLL_FRAMES = 10
OVERLAY_FRAMES = 20
RELIABILITY_FRAMES = 12
THRESHOLD_FRAMES = 36
OUTRO_FRAMES = 16


def _shot(page, frames: list[Image.Image], count: int = 1) -> None:
    """One screenshot, repeated ``count`` times in the frame list."""
    image = Image.open(io.BytesIO(page.screenshot(type="png"))).convert("RGB")
    height = round(image.height * TARGET_WIDTH_PX / image.width)
    resized = image.resize((TARGET_WIDTH_PX, height), Image.LANCZOS)
    frames.extend(resized for _ in range(count))


def _click_slider(page, index: int, fraction: float) -> None:
    """Click one slider at a fraction of its track, which is how a value is set here."""
    slider = page.locator(".q-slider").nth(index)
    box = slider.bounding_box()
    if box is None:
        raise CaptureFailed(f"slider {index} has no bounding box to click in")
    page.mouse.click(box["x"] + box["width"] * fraction, box["y"] + box["height"] / 2)
    page.wait_for_timeout(SETTLE_MS)


def capture_frames(page) -> list[Image.Image]:
    """Drive the scripted interaction and return the frames, in order."""
    frames: list[Image.Image] = []
    page.wait_for_selector("text=Predict", timeout=SELECTOR_TIMEOUT_MS)
    page.wait_for_selector(".js-plotly-plot", timeout=SELECTOR_TIMEOUT_MS)
    page.wait_for_timeout(TAB_SETTLE_MS)
    _shot(page, frames, INTRO_FRAMES)

    # The strength sweep of build spec 15.1: low to high, with the band morphing under it.
    for step in range(SWEEP_FRAMES):
        _click_slider(page, 0, step / (SWEEP_FRAMES - 1))
        _shot(page, frames)
    _shot(page, frames, SWEEP_HOLD_FRAMES)

    # Into the dataset panel, then down to the view the spec calls the most convincing one.
    page.click("text=Dataset")
    page.wait_for_timeout(TAB_SETTLE_MS)
    _shot(page, frames, DATASET_FRAMES)
    for _ in range(SCROLL_FRAMES):
        page.mouse.wheel(0, VIEWPORT_HEIGHT_PX // 2)
        page.wait_for_timeout(SETTLE_MS)
        _shot(page, frames)
    _shot(page, frames, OVERLAY_FRAMES)

    # The reliability threshold slider, recounting the persisted Monte Carlo rows.
    page.mouse.wheel(0, -VIEWPORT_HEIGHT_PX * SCROLL_FRAMES)
    page.wait_for_timeout(SETTLE_MS)
    page.click("text=Reliability")
    page.wait_for_timeout(TAB_SETTLE_MS)
    page.mouse.wheel(0, VIEWPORT_HEIGHT_PX // 2)
    page.wait_for_timeout(SETTLE_MS)
    _shot(page, frames, RELIABILITY_FRAMES)
    for step in range(THRESHOLD_FRAMES):
        _click_slider(page, 0, 0.15 + 0.7 * step / (THRESHOLD_FRAMES - 1))
        _shot(page, frames)
    _shot(page, frames, OUTRO_FRAMES)
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
                frames = capture_frames(page)
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
            "ceiling of build spec 15.1; drop the frame rate or the width and record the "
            "change in docs/DESIGN_DECISIONS.md"
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
