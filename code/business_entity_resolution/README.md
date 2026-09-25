# Business Entity Resolution

`src/simple.py` runs the whole pipeline end to end: reads the data, blocks, scores, predicts, writes both submission files, and scores itself against the training ground truth. Deliberately lean on purpose — a working end-to-end pipeline before refining any stage.

## Setup

```bash
pip install -r requirements.txt
```

The data is not in this repo (`student_resource/` is gitignored, it is >1 GB). Unzip it at the repo root so `student_resource/dataset/train/train_source1.tsv` exists, or pass `--data-root`.

## Run

```bash
cd code/business_entity_resolution

python -m src.simple --split train --sample 200000   # ~1 min smoke test
python -m src.simple --split train                   # full run, prints macro F_0.5
python -m src.simple --split test                    # writes the submission
```

Outputs go to `output/matching_results.tsv` and `output/candidate_pairs.tsv` at the repo root, or wherever `--out-dir` points.

Validate before submitting, from the `student_resource/` directory:

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## How it works

Normalization lowercases, replaces anything non-alphanumeric with a space, and collapses whitespace. Blocking uses a single exact key — country plus the sorted unique tokens of the business name — so `Acme Foods` and `Foods Acme` collide regardless of word order. Keys shared by more than `--max-bucket` records are dropped, since a key that common carries no evidence and would emit a large number of uninformative pairs. Scoring is the Jaccard overlap of address tokens, and prediction keeps pairs at or above `--threshold`, best `--topk` per entity.

Country is folded into the blocking key rather than compared separately, which costs nothing and removes every cross-country comparison.

## Knobs

`--threshold` is the address Jaccard floor and the only dial that moves the score materially. `--topk` caps predictions per entity and is flat above 5, because blocking rarely produces more candidates than that. `--max-bucket` controls how aggressively over-common name keys are discarded; 100 beats both tighter settings tested. `--chunk` sets how many source-1 entities are processed per pass.

`--sample N` matches only the first N Source-1 entities, against the **full** target set, so the reported F_0.5 stays representative — on a quarter of the entities it lands within a few thousandths of the full-run score. `--sample-targets N` additionally truncates Sources 2 and 3 to save memory, and prints a warning, because a Source-1 entity's true partners are scattered anywhere across 10M target rows: keeping only the first N puts most of them out of reach and the score then measures the truncation rather than the pipeline, in either direction.

## Tuned settings

Defaults are `--threshold 0.15 --topk 8 --max-bucket 100`, giving **macro F_0.5 = 0.4670** on the full training split against a do-nothing floor of 0.0558.

The threshold sweep has a clear interior optimum — the address signal is worth keeping, and worth not over-trusting:

| threshold | 0.00 | 0.05 | 0.10 | **0.15** | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| macro F_0.5 | 0.3860 | 0.4281 | 0.4496 | **0.4670** | 0.4664 | 0.4455 | 0.4127 | 0.3672 | 0.3056 | 0.2355 |

`--topk` at 3 / 4 / 5 / 6 scores 0.4447 / 0.4454 / 0.4455 / 0.4455 — flat. `--max-bucket` at 20 / 50 / 100 scores 0.4373 / 0.4505 / 0.4670.

## Memory

Source 1 is streamed in chunks. The resident structure is the source-2/3 key table, roughly 600MB–1GB at full scale. `--sample` runs on a slice for smaller machines.

## Extending it

Each stage is a single clear step, so improvements attach to one stage without changing the shape of the pipeline: richer normalization (abbreviation expansion, transliteration of non-Latin scripts) extends `_clean`, additional blocking keys extend the key built in `load_source`, fuzzy candidate generation extends `build_targets`, and a learned scorer replaces `_jaccard`. Add them one at a time with the F_0.5 number measured before and after.

The `feat/full-pipeline` branch has several of these already built — romanization via IAST, five weighted blocking keys with inverse-frequency scoring, FAISS n-gram candidate search, and per-key recall diagnostics — reaching 64% blocking recall. Its components can be reintroduced here one stage at a time.
