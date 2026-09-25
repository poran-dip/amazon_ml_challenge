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

`--threshold` is the address Jaccard floor and the most useful dial: F_0.5 weights precision twice as heavily as recall, so raising it often pays. `--topk` caps predictions per entity; the true mean is about 3.5 matches, so 5 is a reasonable start. `--max-bucket` controls how aggressively over-common name keys are discarded. `--sample` runs on a slice of each source for a fast iteration loop. `--chunk` sets how many source-1 entities are processed per pass.

## Memory

Source 1 is streamed in chunks. The resident structure is the source-2/3 key table, roughly 600MB–1GB at full scale. `--sample` runs on a slice for smaller machines.

## Extending it

Each stage is a single clear step, so improvements attach to one stage without changing the shape of the pipeline: richer normalization (abbreviation expansion, transliteration of non-Latin scripts) extends `_clean`, additional blocking keys extend the key built in `load_source`, fuzzy candidate generation extends `build_targets`, and a learned scorer replaces `_jaccard`. Add them one at a time with the F_0.5 number measured before and after.

The `feat/full-pipeline` branch has several of these already built — romanization via IAST, five weighted blocking keys with inverse-frequency scoring, FAISS n-gram candidate search, and per-key recall diagnostics — reaching 64% blocking recall. Its components can be reintroduced here one stage at a time.
