"""Paths and shared constants.

Data lives outside the repo (student_resource/ is gitignored). Point the
pipeline at it with the AMLC_DATA env var or the --data-root flag; the default
assumes the provided material was unzipped into <repo_root>/student_resource.
"""

from __future__ import annotations

import os
from pathlib import Path

# .../code/business_entity_resolution/src/config.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DATA_ROOT = REPO_ROOT / "student_resource"
DEFAULT_WORK_ROOT = REPO_ROOT / "work"        # intermediate parquet, caches
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output"    # submission TSVs

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)

# Ground truth only exists for train.
GROUND_TRUTH_NAME = "train_ground_truth.tsv"


def data_root(override: str | os.PathLike | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get("AMLC_DATA")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_DATA_ROOT


def work_root(override: str | os.PathLike | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get("AMLC_WORK")
    root = Path(env).expanduser().resolve() if env else DEFAULT_WORK_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def source_tsv(split: str, source: int, root: Path) -> Path:
    """dataset/<split>/<split>_source<N>.tsv"""
    return root / "dataset" / split / f"{split}_source{source}.tsv"


def ground_truth_tsv(root: Path) -> Path:
    return root / "dataset" / "train" / GROUND_TRUTH_NAME


def normalized_parquet(split: str, source: int, work: Path) -> Path:
    out = work / "normalized"
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{split}_source{source}.parquet"
