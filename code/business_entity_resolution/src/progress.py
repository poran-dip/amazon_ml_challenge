"""Run telemetry: where a run is, how long each stage took, what it cost.

Why this exists
---------------
A full training run takes on the order of two hours and, before this module,
reported almost nothing while it did. Every stage printed a single summary line
*after* it finished, so a run that was going to die at minute 90 on memory
looked identical for the first 89 minutes to a run that was going to succeed.
There was no way to answer "where is it?", "is it making progress?", "will it
fit in 16GB?" or "how much disk has it burned?" without waiting for the end.

Nothing can be optimized that is not measured, and nothing here was measured:
`claude/next-steps.md` records the ~2 hour figure as *inferred from the code,
not timed*, and the per-stage budgets as extrapolations from three numbers
taken under a different candidate count. This module is the instrument that
replaces those guesses.

Two outputs, deliberately separate
----------------------------------
* **Console** (stderr, under `--progress`) — live, human, overwritten in place
  on a terminal. For a person watching a run.
* **`work/timings.jsonl`** (always, unless `--no-timings`) — one JSON object
  per stage, appended across runs forever. For comparing this run against the
  last one. Console output is not a measurement; a file you can diff is.

stderr rather than stdout is deliberate: the pipeline's *results* go to stdout,
so `python -m src.main --progress > run.log` keeps the report clean and leaves
the progress on the terminal, and neither stream has to be parsed out of the
other.

Design constraints
------------------
* **It must never break a run.** Every psutil call, every stat, every write is
  inside a `try`. A telemetry failure at minute 90 of a two-hour run would be a
  far worse bug than the missing telemetry it replaced, so this module has no
  failure mode louder than a missing field in the JSONL.
* **It must be cheap.** The sampler thread wakes twice a second to read one
  number. The directory scan happens at stage boundaries, not in loops. A
  ticker that is not attached to a terminal prints on an interval rather than
  per call, so redirecting to a file does not produce a million lines.
* **It must work on Windows.** That is where the real runs happen. psutil
  covers RSS and IO counters on both platforms; the process-lifetime peak is
  `peak_wset` on Windows and `VmHWM` on Linux, so both are read and whichever
  is available wins.

Usage
-----
Configure once, from the CLI entry point::

    progress.configure(console=a.progress, work=work, argv=sys.argv)

Then wrap work in stages, which nest::

    with progress.stage("blocking", budget=60) as st:
        with progress.stage("merge") as sub:
            t = sub.ticker(n_buckets, unit="buckets")
            for b in range(n_buckets):
                ...
                t.advance(kept=kept)
            t.close()
        st.set(rows=kept)

Every stage records elapsed seconds, the fields it was given, RSS at entry and
exit, the peak RSS *observed inside that stage* by the sampler, the low-water
mark of available system memory, and the bytes written to disk during it —
measured two independent ways (see `_Snapshot`).
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

try:  # pragma: no cover - availability is environmental, not logical
    import psutil
except Exception:  # noqa: BLE001 - psutil is optional by design
    psutil = None  # type: ignore[assignment]

__all__ = [
    "configure",
    "enabled",
    "console_enabled",
    "stage",
    "ticker",
    "note",
    "Ticker",
    "Stage",
]

MB = 1024.0 * 1024.0
GB = 1024.0 * 1024.0 * 1024.0

# The sampler's period. Peak memory inside a single `polars.collect()` is
# invisible at stage boundaries -- the whole point of sampling is to catch a
# spike that begins and ends between two of our own print statements. Half a
# second is short enough to catch a multi-second allocation and long enough to
# be free.
DEFAULT_INTERVAL = 0.5

# How often a non-tty ticker prints. On a terminal the line is rewritten in
# place and can be frequent; redirected to a file it must not be, or a 16-bucket
# merge over 47M rows writes a million lines into the log.
NON_TTY_SECONDS = 30.0


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------

def _hms(seconds: float) -> str:
    """Compact duration. 9.4s / 2m11s / 1h04m.

    Durations here span three orders of magnitude -- a 9 second normalization
    cache hit and a 2 hour training run -- and a bare seconds count is
    unreadable at the top of that range.
    """
    if seconds != seconds or seconds < 0:  # NaN guard
        return "?"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


def _gb(n: float | None) -> str:
    return "?" if n is None else f"{n / GB:.2f}GB"


def _rate(n: float, seconds: float) -> str:
    """Items per second, abbreviated. Guards the zero-elapsed first call."""
    if seconds <= 0:
        return "-"
    r = n / seconds
    if r >= 1e6:
        return f"{r / 1e6:.2f}M/s"
    if r >= 1e3:
        return f"{r / 1e3:.1f}k/s"
    return f"{r:.0f}/s"


# --------------------------------------------------------------------------
# platform probes -- every one of these returns None rather than raising
# --------------------------------------------------------------------------

_PROC = None
if psutil is not None:
    try:
        _PROC = psutil.Process()
    except Exception:  # noqa: BLE001
        _PROC = None


def _rss() -> int | None:
    """Resident set size of this process, in bytes."""
    if _PROC is None:
        return None
    try:
        return int(_PROC.memory_info().rss)
    except Exception:  # noqa: BLE001
        return None


def _peak_rss() -> int | None:
    """Process-LIFETIME peak resident memory, from the OS.

    Distinct from the sampler's per-stage maximum: this one cannot miss a spike
    because the kernel records it, but it also never goes down, so it answers
    "did this run ever come close to 16GB?" and not "which stage did it?". Both
    questions matter, so both numbers are recorded.

    `peak_wset` is Windows-only and `VmHWM` is Linux-only; the run happens on
    the former and this sandbox is the latter, so neither can be assumed.
    """
    if _PROC is not None:
        try:
            info = _PROC.memory_info()
        except Exception:  # noqa: BLE001
            info = None
        if info is not None:
            peak = getattr(info, "peak_wset", None)  # Windows
            if peak:
                return int(peak)
    try:
        with open("/proc/self/status", "r", encoding="ascii", errors="replace") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    # "VmHWM:    123456 kB"
                    return int(line.split()[1]) * 1024
    except Exception:  # noqa: BLE001
        pass
    return None


def _available() -> int | None:
    """Free physical memory, system-wide.

    The interesting failure on a 16GB laptop is not this process's RSS in
    isolation but the machine running out while a browser and an IDE are also
    resident. The low-water mark of this number is what predicts the swap death
    that RSS alone does not.
    """
    if psutil is None:
        return None
    try:
        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001
        return None


def _io() -> tuple[int | None, int | None]:
    """(bytes read, bytes written) by this process since it started."""
    if _PROC is None:
        return (None, None)
    try:
        c = _PROC.io_counters()
    except Exception:  # noqa: BLE001 - AccessDenied and NotImplemented both happen
        return (None, None)
    return (
        int(getattr(c, "read_bytes", 0) or 0),
        int(getattr(c, "write_bytes", 0) or 0),
    )


def _tree_bytes(roots: tuple[Path, ...]) -> int | None:
    """Total size of the artefact directories.

    The second, independent disk measurement. `io_counters().write_bytes` is
    everything the process wrote including rewrites, temporary files and (on
    Windows) non-disk handles; this is what actually survives on disk under
    `work/` and `output/`. A run that writes 40GB to produce 12GB of parquet is
    doing something worth knowing about, and only the pair of numbers shows it.
    """
    total = 0
    seen = False
    for root in roots:
        try:
            if not root.is_dir():
                continue
        except Exception:  # noqa: BLE001
            continue
        seen = True
        stack = [root]
        while stack:
            d = stack.pop()
            try:
                with os.scandir(d) as it:
                    for e in it:
                        try:
                            if e.is_dir(follow_symlinks=False):
                                stack.append(Path(e.path))
                            elif e.is_file(follow_symlinks=False):
                                total += e.stat(follow_symlinks=False).st_size
                        except OSError:
                            continue
            except OSError:
                continue
    return total if seen else None


# --------------------------------------------------------------------------
# snapshots
# --------------------------------------------------------------------------

class _Snapshot:
    """One reading of every cost counter, taken at a stage boundary."""

    __slots__ = ("t", "rss", "peak", "avail", "read", "write", "tree")

    def __init__(self, roots: tuple[Path, ...], scan_tree: bool = True) -> None:
        self.t = time.perf_counter()
        self.rss = _rss()
        self.peak = _peak_rss()
        self.avail = _available()
        self.read, self.write = _io()
        self.tree = _tree_bytes(roots) if scan_tree else None


def _delta(new: int | None, old: int | None) -> int | None:
    if new is None or old is None:
        return None
    return new - old


# --------------------------------------------------------------------------
# sampler
# --------------------------------------------------------------------------

class _Sampler(threading.Thread):
    """Background peak-memory watcher.

    A daemon thread so that an `os._exit` from a watchdog, or an unhandled
    exception on the main thread, can never be held open by telemetry. It only
    ever reads two numbers and takes no lock the main thread waits on: the
    worst a torn read can do is put one stale value in a log line.
    """

    def __init__(self, interval: float) -> None:
        super().__init__(name="amlc-telemetry", daemon=True)
        self.interval = max(0.05, float(interval))
        self._stop = threading.Event()
        self.peak_rss = 0
        self.min_avail: int | None = None

    def run(self) -> None:  # pragma: no cover - timing dependent
        while not self._stop.wait(self.interval):
            r = _rss()
            if r is not None and r > self.peak_rss:
                self.peak_rss = r
            a = _available()
            if a is not None and (self.min_avail is None or a < self.min_avail):
                self.min_avail = a

    def stop(self) -> None:
        self._stop.set()


# --------------------------------------------------------------------------
# the reporter
# --------------------------------------------------------------------------

class _Reporter:
    def __init__(
        self,
        console: bool,
        jsonl: Path | None,
        roots: tuple[Path, ...],
        interval: float,
        run_id: str,
        stream,
    ) -> None:
        self.console = console
        self.jsonl = jsonl
        self.roots = roots
        self.run_id = run_id
        self.stream = stream
        self.stack: list[Stage] = []
        self.t0 = time.perf_counter()
        self.sampler = _Sampler(interval)
        # Only spend a thread if somebody is going to read its output.
        if console or jsonl is not None:
            try:
                self.sampler.start()
            except Exception:  # noqa: BLE001
                pass
        try:
            self.isatty = bool(getattr(stream, "isatty", lambda: False)())
        except Exception:  # noqa: BLE001
            self.isatty = False
        self._line_open = False
        # Length of the transient line currently on screen, so the next one can
        # be padded to cover exactly what it has to erase.
        self._last_len = 0

    # -- console plumbing ---------------------------------------------------

    def width(self) -> int:
        """Usable columns for a line that must NOT wrap.

        Two columns short of the terminal, not one. A line of exactly
        `columns` characters wraps in PowerShell -- the cursor moves to the next
        row and the following `\\r` then returns to the start of the WRONG row,
        which leaves the tail of every long line stranded on screen. Observed on
        the first real run: the normalization ticker left fragments like
        "source1 normalize: 1,750,000/2,20" behind every rewrite.
        """
        if not self.isatty:
            return 10_000
        try:
            return max(40, shutil.get_terminal_size((100, 24)).columns - 2)
        except Exception:  # noqa: BLE001
            return 100

    def write_line(self, text: str, transient: bool = False) -> None:
        """Emit one console line, optionally overwritten by the next one.

        A transient line on a terminal is rewritten in place with `\\r`; off a
        terminal it is a normal line, because `\\r` in a log file is noise.

        A shorter line has to erase the longer one under it, and the erasing is
        done by padding to the PREVIOUS line's length, never to the full
        terminal width. Padding to the width puts the cursor on the last column
        of every single rewrite, which is one character from a wrap, and a wrap
        is what breaks `\\r`.
        """
        if not self.console:
            return
        try:
            if transient and self.isatty:
                w = self.width()
                if len(text) > w:
                    # ASCII only. The console these runs happen on is a Windows
                    # one whose code page is not guaranteed to be UTF-8, and a
                    # UnicodeEncodeError raised from inside telemetry would kill
                    # a two-hour run over a progress line.
                    text = (text[: w - 3] + "...") if w > 3 else text[:w]
                pad = max(self._last_len - len(text), 0)
                self.stream.write("\r" + text + " " * pad)
                self._last_len = len(text)
                self._line_open = True
            else:
                if self._line_open:
                    self.stream.write("\n")
                    self._line_open = False
                    self._last_len = 0
                self.stream.write(text + "\n")
            self.stream.flush()
        except Exception:  # noqa: BLE001
            pass

    def end_transient(self) -> None:
        if self._line_open:
            try:
                self.stream.write("\n")
                self.stream.flush()
            except Exception:  # noqa: BLE001
                pass
            self._line_open = False
            self._last_len = 0

    # -- jsonl --------------------------------------------------------------

    def record(self, payload: dict) -> None:
        if self.jsonl is None:
            return
        try:
            self.jsonl.parent.mkdir(parents=True, exist_ok=True)
            with open(self.jsonl, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, default=str) + "\n")
        except Exception:  # noqa: BLE001
            pass

    def indent(self) -> str:
        return "  " * (len(self.stack) + 1)

    def path(self) -> str:
        return "/".join(s.name for s in self.stack)


_ACTIVE: _Reporter | None = None


def configure(
    console: bool = False,
    work: Path | None = None,
    argv: list[str] | None = None,
    timings: bool = True,
    interval: float = DEFAULT_INTERVAL,
    output: Path | None = None,
    stream=None,
    label: str = "",
) -> None:
    """Install the telemetry for this process. Safe to call more than once.

    `console` is `--progress`. `timings` is on unless `--no-timings`, because
    one appended JSON object per stage costs nothing and the history is the
    thing that makes the next run comparable to this one.

    `AMLC_PROGRESS=1` in the environment turns the console on as well, so a
    stage invoked as its own module (`python -m src.blocking`) inside a script
    does not need the flag threaded through the script.
    """
    global _ACTIVE

    if _ACTIVE is not None:
        # A second configure would start a second sampler thread and leave the
        # first one running forever. `main` configures, then calls into
        # `blocking.block_split`, whose own `main` would configure again if it
        # were ever reached that way -- so this is a real path, not a
        # hypothetical one.
        return

    env = os.environ.get("AMLC_PROGRESS", "").strip().lower()
    if env in {"1", "true", "yes", "on"}:
        console = True

    roots: list[Path] = []
    if work is not None:
        roots.append(Path(work))
    if output is not None:
        roots.append(Path(output))

    jsonl = None
    if timings and work is not None:
        jsonl = Path(work) / "timings.jsonl"

    run_id = time.strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"
    _ACTIVE = _Reporter(
        console=console,
        jsonl=jsonl,
        roots=tuple(roots),
        interval=interval,
        run_id=run_id,
        stream=stream or sys.stderr,
    )

    # Registered rather than called from a `finally` in each entry point.
    # `main` has two early `return`s and several `parser.error` calls, which
    # raise SystemExit, and every one of those paths should still close the run
    # out. atexit covers all of them and `shutdown` is idempotent, so an
    # explicit call at the end of a clean run is not a double-write. The one
    # thing it cannot cover is `os._exit`, which the planned watchdog uses --
    # that is inherent to `os._exit` and is why the per-stage `stage_start`
    # records exist.
    atexit.register(shutdown)

    start = _Snapshot(_ACTIVE.roots)
    _ACTIVE.record({
        "run": run_id,
        "event": "run_start",
        "label": label,
        "wall": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": argv if argv is not None else sys.argv,
        "pid": os.getpid(),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "psutil": getattr(psutil, "__version__", None) if psutil else None,
        "cpu_count": os.cpu_count(),
        "rss_mb": None if start.rss is None else round(start.rss / MB, 1),
        "sys_avail_mb": None if start.avail is None else round(start.avail / MB, 1),
        "sys_total_mb": _total_mb(),
        "tree_gb": None if start.tree is None else round(start.tree / GB, 3),
    })
    if console:
        bits = [f"telemetry on  run={run_id}"]
        tot = _total_mb()
        if tot:
            bits.append(f"ram {tot / 1024:.1f}GB")
        if start.avail is not None:
            bits.append(f"avail {_gb(start.avail)}")
        if start.tree is not None:
            bits.append(f"work+output {_gb(start.tree)}")
        if psutil is None:
            bits.append("psutil MISSING - memory and disk figures disabled "
                        "(pip install psutil)")
        if jsonl is not None:
            bits.append(f"-> {jsonl}")
        _ACTIVE.write_line("[telemetry] " + "  ".join(bits))


def _total_mb() -> float | None:
    if psutil is None:
        return None
    try:
        return round(psutil.virtual_memory().total / MB, 1)
    except Exception:  # noqa: BLE001
        return None


def shutdown() -> None:
    """Stop the sampler and write the run's closing record.

    Not required for correctness -- the thread is a daemon -- but a run that
    ends without it leaves no `run_end` line, and a `timings.jsonl` where the
    last run has no end is indistinguishable from one that was killed.
    """
    global _ACTIVE
    r = _ACTIVE
    if r is None:
        return
    r.end_transient()
    snap = _Snapshot(r.roots)
    r.record({
        "run": r.run_id,
        "event": "run_end",
        "seconds": round(snap.t - r.t0, 2),
        "rss_mb": None if snap.rss is None else round(snap.rss / MB, 1),
        "peak_rss_mb": _peak_mb(r, snap),
        "min_sys_avail_mb": (
            None if r.sampler.min_avail is None
            else round(r.sampler.min_avail / MB, 1)
        ),
        "io_write_gb": None if snap.write is None else round(snap.write / GB, 3),
        "tree_gb": None if snap.tree is None else round(snap.tree / GB, 3),
    })
    if r.console:
        r.write_line(
            f"[telemetry] run {r.run_id} finished in {_hms(snap.t - r.t0)}   "
            f"peak rss {_gb(_peak_bytes(r, snap))}   "
            f"disk written {_gb(snap.write)}   "
            f"work+output now {_gb(snap.tree)}"
        )
    try:
        r.sampler.stop()
    except Exception:  # noqa: BLE001
        pass
    _ACTIVE = None


def _peak_bytes(r: _Reporter, snap: _Snapshot) -> int | None:
    """The larger of the OS lifetime peak and the sampler's observed maximum."""
    cands = [v for v in (snap.peak, r.sampler.peak_rss or None) if v]
    return max(cands) if cands else None


def _peak_mb(r: _Reporter, snap: _Snapshot) -> float | None:
    b = _peak_bytes(r, snap)
    return None if b is None else round(b / MB, 1)


def enabled() -> bool:
    """True if anything at all is being recorded."""
    return _ACTIVE is not None


def console_enabled() -> bool:
    """True if live console output is on -- i.e. `--progress` was passed.

    Call sites use this to decide whether to compute a number that is only
    worth computing in order to print it.
    """
    return _ACTIVE is not None and _ACTIVE.console


def note(message: str, **fields) -> None:
    """A one-off line, printed under `--progress` and always recorded."""
    r = _ACTIVE
    if r is None:
        return
    r.record({
        "run": r.run_id,
        "event": "note",
        "stage": r.path(),
        "message": message,
        "elapsed": round(time.perf_counter() - r.t0, 2),
        **fields,
    })
    if r.console:
        extra = "  ".join(f"{k}={_fmt_field(v)}" for k, v in fields.items())
        r.write_line(f"{r.indent()}- {message}" + (f"   {extra}" if extra else ""))


def _fmt_field(v) -> str:
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.3f}"
    return str(v)


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

class Stage:
    """One timed, nested unit of work. Created by `stage()`, not directly."""

    def __init__(self, reporter: _Reporter, name: str, fields: dict) -> None:
        self._r = reporter
        self.name = name
        self.fields = dict(fields)
        self.start = _Snapshot(reporter.roots)
        # The sampler's peak is monotone over the process, so a per-stage
        # maximum has to be measured as a delta from the value on entry rather
        # than read off the counter.
        self._peak_at_entry = reporter.sampler.peak_rss
        self._tickers: list[Ticker] = []
        # Captured when the stage is opened, because `_close` runs AFTER the
        # stage has been popped off the stack: reading them from the live stack
        # there would report the parent's path, the parent's depth, and an
        # indent one level shallower than the stage's own opening line.
        self.path = name
        self.depth = 0
        self._indent = "  "

    # -- mutation -----------------------------------------------------------

    def set(self, **fields) -> None:
        """Attach or overwrite fields recorded when the stage closes.

        Row counts are usually only known at the end, so they cannot be passed
        to `stage()`.
        """
        self.fields.update(fields)

    def note(self, message: str, **fields) -> None:
        note(message, **fields)

    def ticker(self, total: int | None, unit: str = "rows", label: str = "") -> "Ticker":
        t = Ticker(self._r, label or self.name, total, unit)
        self._tickers.append(t)
        return t

    # -- lifecycle ----------------------------------------------------------

    def _open(self) -> None:
        r = self._r
        # Called with `self` already on the stack, so these are the stage's own
        # coordinates rather than its parent's.
        self.path = r.path()
        self.depth = max(len(r.stack) - 1, 0)
        self._indent = r.indent()
        if r.console:
            head = f"{self._indent}> {self.name}"
            if self.fields:
                head += "   " + "  ".join(
                    f"{k}={_fmt_field(v)}" for k, v in self.fields.items()
                )
            r.write_line(head)
        # Recorded as well as printed, so `timings.jsonl` shows how far a run
        # that never finished actually got. A stage that only writes a record
        # when it closes leaves nothing at all behind when the run is killed --
        # which is precisely the run whose timings matter most.
        r.record({
            "run": r.run_id,
            "event": "stage_start",
            "stage": self.name,
            "path": self.path,
            "depth": self.depth,
            "elapsed": round(self.start.t - r.t0, 2),
            "rss_mb": None if self.start.rss is None else round(self.start.rss / MB, 1),
            "sys_avail_mb": (
                None if self.start.avail is None
                else round(self.start.avail / MB, 1)
            ),
            **{k: v for k, v in self.fields.items()},
        })

    def _close(self, failed: BaseException | None) -> None:
        r = self._r
        for t in self._tickers:
            t.close(final=False)
        end = _Snapshot(r.roots)
        seconds = end.t - self.start.t
        # The sampler's counter is monotone over the whole process, so a
        # per-stage peak is only meaningful if it ROSE during the stage.
        # Otherwise the honest answer for this stage is its own endpoint RSS,
        # not a high-water mark inherited from an earlier stage -- reporting the
        # latter would make a cheap stage look like the expensive one.
        observed = r.sampler.peak_rss
        # The endpoints are always a valid lower bound on the stage's own peak,
        # and on a stage shorter than the sampler's period they are the ONLY
        # evidence -- without them a fast stage reported a "peak" below its own
        # closing RSS, which is nonsense on its face.
        floor = max([v for v in (self.start.rss, end.rss) if v] or [0])
        stage_peak: int | None = floor or None
        if observed and observed > self._peak_at_entry:
            stage_peak = max(observed, floor) or None
        io_written = _delta(end.write, self.start.write)
        tree_grew = _delta(end.tree, self.start.tree)

        payload = {
            "run": r.run_id,
            "event": "stage",
            "stage": self.name,
            "path": self.path,
            "depth": self.depth,
            "seconds": round(seconds, 3),
            "ok": failed is None,
            "rss_start_mb": None if self.start.rss is None else round(self.start.rss / MB, 1),
            "rss_end_mb": None if end.rss is None else round(end.rss / MB, 1),
            "rss_peak_mb": None if stage_peak is None else round(stage_peak / MB, 1),
            "sys_avail_end_mb": None if end.avail is None else round(end.avail / MB, 1),
            "io_write_gb": None if io_written is None else round(io_written / GB, 3),
            "tree_delta_gb": None if tree_grew is None else round(tree_grew / GB, 3),
            "tree_gb": None if end.tree is None else round(end.tree / GB, 3),
            **{k: v for k, v in self.fields.items()},
        }
        if failed is not None:
            payload["error"] = f"{type(failed).__name__}: {failed}"
        r.record(payload)

        if r.console:
            bits = [f"{_hms(seconds)}"]
            rows = self.fields.get("rows")
            if isinstance(rows, (int, float)) and rows:
                bits.append(f"{int(rows):,} rows ({_rate(float(rows), seconds)})")
            for k, v in self.fields.items():
                if k == "rows":
                    continue
                bits.append(f"{k}={_fmt_field(v)}")
            if end.rss is not None:
                bits.append(f"rss {_gb(end.rss)}")
            if stage_peak is not None:
                bits.append(f"peak {_gb(stage_peak)}")
            if end.avail is not None:
                bits.append(f"avail {_gb(end.avail)}")
            if io_written:
                bits.append(f"wrote {_gb(io_written)}")
            if tree_grew:
                bits.append(f"artefacts {tree_grew / GB:+.2f}GB")
            mark = "<" if failed is None else "x"
            r.write_line(f"{self._indent}{mark} {self.name}   " + "  ".join(bits))


@contextmanager
def stage(name: str, **fields):
    """Time a block of work, nested inside any enclosing stage.

    Yields a `Stage` when telemetry is on and a no-op stand-in when it is not,
    so call sites never have to branch on whether `--progress` was passed.
    """
    r = _ACTIVE
    if r is None:
        yield _NullStage()
        return
    st = Stage(r, name, fields)
    r.stack.append(st)
    st._open()
    failed: BaseException | None = None
    try:
        yield st
    except BaseException as exc:  # noqa: BLE001 - re-raised below
        failed = exc
        raise
    finally:
        # Pop BEFORE closing so the closing line is indented at the stage's own
        # level rather than its children's.
        try:
            r.stack.pop()
        except IndexError:
            pass
        st._close(failed)


class _NullStage:
    """What `stage()` yields when telemetry is off. Every method does nothing."""

    def set(self, **fields) -> None:
        pass

    def note(self, message: str, **fields) -> None:
        pass

    def ticker(self, total: int | None, unit: str = "rows", label: str = "") -> "Ticker":
        return _NULL_TICKER


# --------------------------------------------------------------------------
# tickers
# --------------------------------------------------------------------------

class Ticker:
    """A live counter for a loop: position, rate, ETA, memory, disk.

    The ETA is a straight-line extrapolation of the mean rate so far, which is
    honest about being an estimate and is right about the order of magnitude --
    the question it answers is "ten more minutes or two more hours?", and no
    loop here has a wildly non-uniform cost per item.
    """

    def __init__(self, reporter: _Reporter, label: str, total: int | None, unit: str) -> None:
        self._r = reporter
        self.label = label
        self.total = total if (total and total > 0) else None
        self.unit = unit
        self.done = 0
        self.t0 = time.perf_counter()
        self._last_emit = 0.0
        self._closed = False
        # Fixed at construction. Read at render time instead, the line would
        # shift sideways whenever an unrelated nested stage opened or closed
        # while the loop was running.
        self._indent = reporter.indent()

    def advance(self, n: int = 1, **extra) -> None:
        self.set(self.done + n, **extra)

    def set(self, done: int, **extra) -> None:
        self.done = int(done)
        self._maybe_emit(extra)

    def _maybe_emit(self, extra: dict) -> None:
        if self._closed or not self._r.console:
            return
        now = time.perf_counter()
        # Off a terminal, print on an interval instead of on every call.
        if not self._r.isatty and (now - self._last_emit) < NON_TTY_SECONDS:
            return
        # On a terminal, still cap the rewrite rate: a per-chunk loop can call
        # this thousands of times a second and flushing stderr is not free.
        if self._r.isatty and (now - self._last_emit) < 0.2:
            return
        self._last_emit = now
        self._r.write_line(self._render(now, extra), transient=True)

    def _render(self, now: float, extra: dict) -> str:
        el = now - self.t0
        bits = []
        if self.total:
            bits.append(f"{self.done:,}/{self.total:,} {self.unit}")
            bits.append(f"{self.done / self.total:5.1%}")
        else:
            bits.append(f"{self.done:,} {self.unit}")
        bits.append(_rate(self.done, el))
        bits.append(f"el {_hms(el)}")
        if self.total and self.done > 0:
            remaining = (self.total - self.done) * (el / self.done)
            bits.append(f"eta {_hms(max(remaining, 0.0))}")
        for k, v in extra.items():
            bits.append(f"{k} {_fmt_field(v)}")
        r = _rss()
        if r is not None:
            bits.append(f"rss {_gb(r)}")
        # The sampler's counter lags by up to its period, so on a fast loop it
        # can sit BELOW the reading taken on the line above -- a "peak" smaller
        # than the current value, which reads as a bug. The current reading is
        # itself a peak observation, so it belongs in the max.
        peak = max(self._r.sampler.peak_rss, r or 0)
        if peak:
            bits.append(f"peak {_gb(peak)}")
        avail = _available()
        if avail is not None:
            bits.append(f"avail {_gb(avail)}")
        _, w = _io()
        if w:
            bits.append(f"wrote {_gb(w)}")
        return f"{self._indent}  {self.label}: " + "  ".join(bits)

    def close(self, final: bool = True) -> None:
        """Finish the line. Idempotent, so a `finally` and an explicit call
        cannot both emit it."""
        if self._closed:
            return
        self._closed = True
        if not self._r.console:
            return
        if final:
            now = time.perf_counter()
            self._r.write_line(self._render(now, {}), transient=True)
        self._r.end_transient()


class _NullTicker:
    def advance(self, n: int = 1, **extra) -> None:
        pass

    def set(self, done: int, **extra) -> None:
        pass

    def close(self, final: bool = True) -> None:
        pass


_NULL_TICKER = _NullTicker()


def ticker(total: int | None, unit: str = "rows", label: str = "") -> Ticker | _NullTicker:
    """A ticker outside any stage, for a loop in a function that has no stage."""
    r = _ACTIVE
    if r is None:
        return _NULL_TICKER
    return Ticker(r, label or "progress", total, unit)


# --------------------------------------------------------------------------
# CLI wiring
# --------------------------------------------------------------------------

def add_arguments(p) -> None:
    """Add the telemetry flags to an argparse parser.

    Shared by every entry point (`main`, `blocking`, `features`, `model`,
    `block_eval`) so the flags cannot drift apart between them.
    """
    g = p.add_argument_group("telemetry")
    g.add_argument(
        "--progress", action="store_true",
        help="Live per-stage and per-chunk progress on stderr: position, rate, "
             "ETA, RSS, peak RSS, free system memory and bytes written. "
             "Results still go to stdout, so `--progress > run.log` separates "
             "them. AMLC_PROGRESS=1 does the same.",
    )
    g.add_argument(
        "--progress-interval", type=float, default=DEFAULT_INTERVAL,
        help="Seconds between memory samples (default: %(default)s). The "
             "sampler is what catches a peak that begins and ends inside one "
             "polars call.",
    )
    g.add_argument(
        "--no-timings", action="store_true",
        help="Do not append per-stage records to work/timings.jsonl. The file "
             "is the run-to-run comparison, so this is rarely what you want.",
    )


def configure_from_args(a, work: Path | None, output: Path | None = None,
                        label: str = "") -> None:
    """Install telemetry from a parsed argparse namespace."""
    configure(
        console=bool(getattr(a, "progress", False)),
        work=work,
        output=output,
        argv=sys.argv,
        timings=not bool(getattr(a, "no_timings", False)),
        interval=float(getattr(a, "progress_interval", DEFAULT_INTERVAL)),
        label=label,
    )
