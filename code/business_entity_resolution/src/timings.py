"""Read `work/timings.jsonl` back: where a run spent its time, memory and disk.

`src.progress` writes the records; this reads them. Separate because writing
happens inside a two-hour run that must not be slowed down or broken, and
reading happens afterwards, interactively, and can afford to be chatty.

    python -m src.timings                  # the last run, slowest stage first
    python -m src.timings --runs 3         # the last three, side by side
    python -m src.timings --list           # every run in the file
    python -m src.timings --tree           # nesting rather than a ranking

What to read it for
-------------------
* **The ranking.** `next-steps.md` records the ~2 hour figure as inferred from
  the code rather than timed, and the per-stage budgets as extrapolations. The
  ranking replaces both with measurements, and the top line is where an
  optimization is worth writing.
* **`self` time.** A parent stage's total includes its children, so a ranking
  by total time puts `model` at the top of every run and says nothing. `self`
  is total minus the children's, which is the time the stage itself spent.
* **The memory column.** The peak per stage says which stage a 16GB machine
  dies in, and `avail` says how close the machine as a whole came.
* **The disk columns.** `write` is what the process wrote, including rewrites;
  `artefacts` is the net change in `work/` and `output/`. A stage that writes
  far more than it leaves behind is doing redundant IO.
* **The comparison.** Two runs of the same stage set, with one flag changed, is
  how a speedup is attributed rather than assumed.

Nothing here is required by the pipeline: this module is read-only over a log
file and importing it has no side effects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import paths


def load(path: Path) -> list[dict]:
    """Every record in the file, in order.

    A partial last line is skipped rather than raised on: the file is appended
    to by a run that may have been killed mid-write, and refusing to read the
    log of the run you most want to look at would be exactly backwards.
    """
    out: list[dict] = []
    if not path.is_file():
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    return out


def runs(records: list[dict]) -> list[str]:
    """Run ids in first-seen order."""
    seen: list[str] = []
    for r in records:
        rid = r.get("run")
        if rid and rid not in seen:
            seen.append(rid)
    return seen


def _f(v, fmt: str = "{:.2f}", dash: str = "-") -> str:
    return dash if v is None else fmt.format(v)


def _self_seconds(stages: list[dict]) -> dict[int, float]:
    """Each stage's time minus its direct children's, keyed by list index.

    Children are identified by path prefix, and only the NEXT depth down is
    subtracted -- subtracting every descendant would double-count a
    grandchild, which is already inside its parent's total.

    Ordering matters: the records are appended when a stage CLOSES, so a parent
    always appears after its children. A child is therefore any earlier record
    one level deeper whose path starts with this one's, and it cannot be
    confused with a same-named stage from a later part of the run.
    """
    self_s: dict[int, float] = {}
    for i, s in enumerate(stages):
        total = float(s.get("seconds") or 0.0)
        path = s.get("path") or s.get("stage") or ""
        depth = int(s.get("depth") or 0)
        claimed = 0.0
        for j in range(i - 1, -1, -1):
            c = stages[j]
            cpath = c.get("path") or ""
            if int(c.get("depth") or 0) != depth + 1:
                continue
            if cpath.startswith(path + "/"):
                claimed += float(c.get("seconds") or 0.0)
        self_s[i] = max(total - claimed, 0.0)
    return self_s


def summarize(records: list[dict], run_id: str, tree: bool, top: int) -> None:
    mine = [r for r in records if r.get("run") == run_id]
    start = next((r for r in mine if r.get("event") == "run_start"), {})
    end = next((r for r in mine if r.get("event") == "run_end"), None)
    stages = [r for r in mine if r.get("event") == "stage"]
    notes = [r for r in mine if r.get("event") == "note"]
    started = [r for r in mine if r.get("event") == "stage_start"]

    argv = start.get("argv") or []
    print("=" * 96)
    print(f"run {run_id}   {start.get('wall', '?')}   {start.get('label', '')}")
    if argv:
        print("  " + " ".join(str(a) for a in argv))
    ram = start.get("sys_total_mb")
    print(f"  {start.get('platform', '?')}  python {start.get('python', '?')}"
          + (f"  ram {ram / 1024:.1f}GB" if ram else "")
          + f"  cpus {start.get('cpu_count', '?')}")
    if end is None:
        # The single most useful thing this tool can tell you, so it is said
        # before the numbers rather than after them.
        unclosed = {s.get("path") for s in started} - {s.get("path") for s in stages}
        print("  *** NO run_end RECORD: this run was killed or is still going."
              + (f"  Unfinished: {', '.join(sorted(p for p in unclosed if p))}"
                 if unclosed else ""))
    else:
        print(f"  total {end.get('seconds')}s"
              f"   peak rss {_f(end.get('peak_rss_mb'), '{:,.0f}')}MB"
              f"   min free ram {_f(end.get('min_sys_avail_mb'), '{:,.0f}')}MB"
              f"   disk written {_f(end.get('io_write_gb'), '{:.2f}')}GB"
              f"   artefacts now {_f(end.get('tree_gb'), '{:.2f}')}GB")
    if not stages:
        print("  (no completed stages)")
        return

    self_s = _self_seconds(stages)
    order = (
        range(len(stages)) if tree
        else sorted(range(len(stages)), key=lambda i: -self_s[i])
    )
    head = (f"  {'stage':<40} {'total':>9} {'self':>9} {'rows':>13} "
            f"{'peak MB':>9} {'write GB':>9} {'artefact GB':>12}")
    print()
    print(head)
    print("  " + "-" * (len(head) - 2))
    shown = 0
    for i in order:
        s = stages[i]
        if not tree and top and shown >= top:
            break
        shown += 1
        depth = int(s.get("depth") or 0)
        label = ("  " * depth + str(s.get("stage"))) if tree else str(s.get("path"))
        if len(label) > 40:
            label = label[:37] + "..."
        rows = s.get("rows")
        flag = "" if s.get("ok", True) else "  FAILED"
        print(f"  {label:<40} {_f(s.get('seconds'), '{:8.2f}s'):>9} "
              f"{self_s[i]:8.2f}s {('' if rows is None else f'{int(rows):,}'):>13} "
              f"{_f(s.get('rss_peak_mb'), '{:,.0f}'):>9} "
              f"{_f(s.get('io_write_gb'), '{:.2f}'):>9} "
              f"{_f(s.get('tree_delta_gb'), '{:+.2f}'):>12}{flag}")
        if not s.get("ok", True) and s.get("error"):
            print(f"      {s['error']}")

    if notes:
        print("\n  notes:")
        for n in notes:
            extra = "  ".join(
                f"{k}={v}" for k, v in n.items()
                if k not in {"run", "event", "stage", "message", "elapsed"}
            )
            print(f"    [{_f(n.get('elapsed'), '{:7.1f}')}s] {n.get('message')}"
                  + (f"   {extra}" if extra else ""))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Summarize work/timings.jsonl: the slowest stages, their "
                    "memory peaks and their disk cost."
    )
    p.add_argument("--work-root", default=None)
    p.add_argument("--file", default=None,
                   help="Path to a timings.jsonl. Defaults to "
                        "<work-root>/timings.jsonl.")
    p.add_argument("--runs", type=int, default=1,
                   help="How many of the most recent runs to summarize "
                        "(default: %(default)s).")
    p.add_argument("--run", default=None,
                   help="A specific run id, as --list prints it.")
    p.add_argument("--list", action="store_true",
                   help="List the runs in the file and stop.")
    p.add_argument("--tree", action="store_true",
                   help="Show stages nested in execution order instead of "
                        "ranked by self time.")
    p.add_argument("--top", type=int, default=25,
                   help="Stages to show in the ranking; 0 for all "
                        "(default: %(default)s).")
    a = p.parse_args()

    path = (
        Path(a.file).expanduser() if a.file
        else paths.work_root(a.work_root) / "timings.jsonl"
    )
    records = load(path)
    if not records:
        print(f"{path} is missing or empty — run the pipeline once without "
              "--no-timings.")
        return

    ids = runs(records)
    if a.list:
        print(f"{len(ids)} run(s) in {path}:")
        for rid in ids:
            st = next((r for r in records
                       if r.get("run") == rid and r.get("event") == "run_start"), {})
            en = next((r for r in records
                       if r.get("run") == rid and r.get("event") == "run_end"), None)
            n = sum(1 for r in records
                    if r.get("run") == rid and r.get("event") == "stage")
            print(f"  {rid}  {st.get('wall', '?')}  {st.get('label', ''):<18} "
                  f"{n:>3} stages  "
                  + (f"{en.get('seconds')}s" if en else "INCOMPLETE"))
        return

    chosen = [a.run] if a.run else ids[-max(a.runs, 1):]
    for rid in chosen:
        if rid not in ids:
            print(f"no such run: {rid}")
            continue
        summarize(records, rid, a.tree, a.top)
        print()


if __name__ == "__main__":
    main()
