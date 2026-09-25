"""Where everything lives.

Split out of `main` so that `blocking` and `block_eval` can locate the
normalized parquet without importing the pipeline that writes it, which would
be circular.

The provided material is never committed (`student_resource/` is gitignored).
Point the pipeline at it with `AMLC_DATA` or `--data-root`; the default assumes
it was unzipped into `<repo_root>/student_resource`. Intermediates go to
`<repo_root>/work` unless `AMLC_WORK` or `--work-root` says otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

# .../code/business_entity_resolution/src/paths.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DATA_ROOT = REPO_ROOT / "student_resource"
DEFAULT_WORK_ROOT = REPO_ROOT / "work"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output"

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def data_root(override: str | os.PathLike | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get("AMLC_DATA")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_DATA_ROOT


def work_root(override: str | os.PathLike | None = None) -> Path:
    if override:
        root = Path(override).expanduser().resolve()
    else:
        env = os.environ.get("AMLC_WORK")
        root = Path(env).expanduser().resolve() if env else DEFAULT_WORK_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def output_root(override: str | os.PathLike | None = None) -> Path:
    """Both splits default here, so `--split test` must run LAST or get its own
    `--out-dir`. A training run afterwards silently overwrites the submission
    with training predictions."""
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_OUTPUT_ROOT


def source_tsv(split: str, source: int, data: Path) -> Path:
    return data / "dataset" / split / f"{split}_source{source}.tsv"


def ground_truth_tsv(data: Path) -> Path:
    """Only exists for train."""
    return data / "dataset" / "train" / "train_ground_truth.tsv"


def _sub(work: Path, name: str) -> Path:
    out = work / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def normalized_parquet(split: str, source: int, work: Path) -> Path:
    return _sub(work, "normalized") / f"{split}_source{source}.parquet"


def partial_parquet(split: str, key: str, work: Path) -> Path:
    """One blocking key's raw hits, before the merge."""
    return _sub(work, "candidates") / f"{split}_partial_{key}.parquet"


def candidates_parquet(split: str, work: Path) -> Path:
    return _sub(work, "candidates") / f"{split}_candidates.parquet"
