# Business Entity Resolution — pipeline

Reproducible pipeline for the ML Challenge 2026 entity resolution task.

## Setup

```bash
pip install -r requirements.txt
```

The provided material is **not** in this repo (`student_resource/` is gitignored because the data is >1 GB). Unzip it at the repo root so that `student_resource/dataset/train/train_source1.tsv` exists, or point the pipeline elsewhere:

```bash
export AMLC_DATA=/path/to/student_resource   # folder containing dataset/
export AMLC_WORK=/path/to/scratch            # intermediates; defaults to <repo>/work
```

All commands below run from this directory (`code/business_entity_resolution/`).

## Stages

Each stage reads the previous stage's parquet and writes its own. Nothing after stage 1 ever touches the raw TSVs.

### 1. Normalize

```bash
python -m src.normalize --split train
python -m src.normalize --split test
```

Writes `$AMLC_WORK/normalized/<split>_source<N>.parquet`. Processes in chunks (`--batch-size`, default 500k rows) so peak memory stays flat regardless of file size. Use `--sample 50000` while iterating on rules — a full pass is only worth paying for once the rules have stopped changing.

Output columns keep both the raw and normalized forms of each field, plus:

| column                    | what it is                                                                              | why                                |
| ------------------------- | --------------------------------------------------------------------------------------- | ---------------------------------- |
| `name_norm`               | cleaned name, legal abbreviations expanded                                              | general-purpose                    |
| `name_core`               | `name_norm` minus legal-form tokens                                                     | `Acme Corp` == `Acme Corporation`  |
| `name_sorted`             | unique tokens, sorted                                                                   | word-order transpositions          |
| `name_nospace`            | `name_core` with spaces removed                                                         | `SunRise` == `Sun Rise`            |
| `name_acronym`            | first letter of each token                                                              | links `IBM` to the expanded form   |
| `addr_norm` / `addr_core` | cleaned address, street abbreviations expanded, locality stopwords dropped from `_core` | `Rd` == `Road`                     |
| `addr_sorted`             | unique non-numeric tokens, sorted                                                       | component reordering               |
| `addr_numbers`            | every digit run                                                                         | house/plot numbers are high-signal |
| `addr_postal`             | digit runs of length 4–6                                                                | generic postal-code candidates     |

### 0. EDA (run this before touching the rules again)

```bash
python -m src.eda --sample 200000
```

Reports per-source profiles, blocking-key selectivity, and — the number that matters — **matched-pair agreement**: for each normalized field, the fraction of _true_ ground-truth pairs whose values are identical. That is the recall ceiling for any exact-match blocking key on that field, so it is the metric to move when changing a rule. It also prints true pairs that normalization failed to collapse, which is the rule backlog.

Later stages (blocking, features, training, prediction) land here as they are built.

## Design notes

**Country is an open set.** The test set contains France, which never appears in training. Nothing in the pipeline branches on a country value, enumerates `{US, India}`, or one-hot encodes country. Postal codes are extracted with a generic 4–6 digit-run rule rather than per-country regexes.

**Normalization is deliberately conservative.** Stage 1 only collapses variants that are _always_ equivalent (`Rd`/`Road`, `Pvt`/`Private`). Judgement calls — whether a landmark reference matters, how much a shared street name is worth — are left to the feature stage, where a model can weigh them, rather than having the rules destroy the information first.

**Vectorized, no Python UDFs.** All normalization is polars string expressions. At ~10M rows per split a per-row Python call would dominate runtime.

**Known gap:** region/state abbreviations (`IL` vs `Illinois`, and equivalents in other countries) are not expanded yet. Whether that is worth a lookup table is a question for the EDA output, not a guess.

## Rule iteration loop

1. Edit a table in `src/text_norm.py` (they are plain dicts — the rules are data, not code).
2. `python -m src.normalize --split train --sample 200000`
3. `python -m src.eda --sample 200000`
4. Compare matched-pair agreement against the previous run; keep the change only if it went up.
