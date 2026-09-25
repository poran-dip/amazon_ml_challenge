# Business Entity Resolution

`src/main.py` is the pipeline and the single entry point for every run: it reads the data, normalizes, blocks, scores, predicts, writes both submission files, and scores itself against the training ground truth. Every submission comes from here, and every improvement lands here.

## Setup

```bash
pip install -r requirements.txt
```

The data is not in this repo (`student_resource/` is gitignored, it is >1 GB). Unzip it at the repo root so `student_resource/dataset/train/train_source1.tsv` exists, or pass `--data-root`.

## Run

```bash
cd code/business_entity_resolution

python -m src.main --split train --sample 200000   # ~1 min smoke test
python -m src.main --split train                   # full run, prints macro F_0.5 + diagnostics
python -m src.main --split test                    # writes the submission
```

Outputs go to `output/matching_results.tsv` and `output/candidate_pairs.tsv` at the repo root, or wherever `--out-dir` points. **Both splits write to the same directory**, so run `--split test` last (or give it its own `--out-dir`) — a training run afterwards overwrites the submission with training predictions.

Validate before submitting, from the `student_resource/` directory:

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## How it works

Five stages, each one function with an explicit input and output.

**Normalize** (`prepare`, rules in `text_norm.py`) folds accents and IAST diacritics, glues runs of single letters into one token so `S.B.I. ATM` does not become `south b i atm`, expands legal and street abbreviations by whole-token lookup, and strips pure legal-form markers to produce `name_core`. Rule ordering is load-bearing: initialism gluing must precede abbreviation expansion. Indic romanization (`translit.py`) is available behind `--translit` but off by default; see *Tuned settings*.

Accent handling deliberately strips only the Latin combining-diacritics block, never `\p{M}`. In Indic scripts the vowel signs and virama are themselves Mark-category characters, so a blanket strip deletes the vowels.

**Block** (`build_targets`) uses a single exact key — country plus the sorted unique tokens of `name_core` — so `Acme Foods` and `Foods Acme` collide regardless of word order. Keys shared by more than `--max-bucket` records are dropped, since a key that common carries almost no evidence. Country is folded into the key rather than compared separately: it agrees on 100% of true training pairs, so it is a free hard partition. It is treated as an opaque label and never enumerated, because the test set contains France and the training set does not.

**Score** (`_jaccard`) is the token-set overlap of `addr_core`. **Predict** (`generate`) keeps pairs at or above `--threshold`, best `--topk` per entity. **Evaluate** (`macro_f05`) reports macro F_0.5 plus macro precision, macro recall, the share of entities given an empty prediction, how many of those have real matches, and mean predicted versus true count.

## Knobs

`--threshold` is the address Jaccard floor and currently the dial that moves the score most. `--topk` caps predictions per entity. `--max-bucket` controls how aggressively over-common keys are discarded. `--chunk` sets how many source-1 entities are processed per pass.

`--sample N` matches only the first N Source-1 entities, against the **full** target set, so the reported F_0.5 stays representative — on a quarter of the entities it lands within a few thousandths of the full-run score. `--sample-targets N` additionally truncates Sources 2 and 3 to save memory, and prints a warning, because a Source-1 entity's true partners are scattered anywhere across 10M target rows: keeping only the first N puts most of them out of reach and the score then measures the truncation rather than the pipeline, in either direction.

`--legacy-norm` reproduces the superseded normalization bit-for-bit. It is a measurement reference for attributing a regression to the normalization stage, not a supported configuration.

## Tuned settings

Defaults are `--threshold 0.15 --topk 8 --max-bucket 100`. Macro F_0.5 on the full training split, against a do-nothing floor of 0.0558:

| normalization | macro F_0.5 | candidates | runtime |
| --- | --- | --- | --- |
| superseded (`--legacy-norm`) | 0.4670 | 14.0M | 27s |
| current | **0.6070** | 27.1M | 176s |
| current + `--translit` | 0.6075 | 27.4M | 302s |

Romanization is off by default: +0.0005 for +126s. Exact-key blocking requires *every* token to agree, and a partly-romanized name still differs on its Latin tokens. Its measured value is character-trigram similarity on cross-script pairs (0.000 → 0.326), which only pays once a fuzzy stage exists — so the code stays.

### What is stale

`--threshold 0.15` and `--max-bucket 100` were both tuned against the superseded normalization and are now mis-tuned by an unmeasured amount. `addr_core` expands abbreviations and drops stopwords, so the Jaccard distribution has shifted. The threshold sweep below is retained for reference but **no longer describes the current pipeline**:

| threshold | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| macro F_0.5 (superseded norm) | 0.3860 | 0.4281 | 0.4496 | 0.4670 | 0.4664 | 0.4455 | 0.4127 | 0.3672 | 0.3056 | 0.2355 |

The precision and recall figures printed after each training run indicate which direction the threshold should move, which a single F_0.5 value cannot.

## Scale

Roughly 12.5M rows per split. Sources are read once and normalized in slices; the resident structure is the source-2/3 key table, about 600MB–1GB at full scale, which fits in 16GB alongside one slice.

The reader avoids two tempting APIs on purpose. `scan_csv(...).slice(offset, chunk).collect()` in a loop re-reads the file from the start every time — 42 passes over a 10M-row TSV. `read_csv_batched` is deprecated in current polars, and its replacement `LazyFrame.collect_batches` does not exist in 1.18, is documented as unstable, and warns that it is slower than the alternatives. Reading once and slicing the DataFrame is zero-copy and version-agnostic; outputs are verified identical on polars 1.18.0 and 1.44.2.

## Working on it

One change at a time, measured against the run it replaces, with the previous behaviour reachable behind a flag rather than deleted. That is what makes a regression attributable instead of something to bisect.

The stage boundaries are where the next changes go. A second blocking key extends `build_targets` — measured per-key diagnostics put `addr_sorted` first, with the highest unique contribution of any key (11.2%, more than every name key combined) despite the weakest agreement (23.5%). A fuzzy candidate stage appends to the candidate set before `generate` filters it, targeting the 21.4% of true pairs that no exact key reaches at any cap. A learned scorer replaces `_jaccard` and the threshold comparison with a model over several pair features; nothing has been trained yet, so no precision figure from a model exists.

`claude/findings.md` in the project holds the measured numbers behind all of the above. The `feat/full-pipeline` branch has working implementations of romanization, five weighted blocking keys with inverse-frequency scoring, FAISS n-gram search and per-key recall diagnostics; its components can be reintroduced here one stage at a time.
