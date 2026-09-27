# Business Entity Resolution — Pyroflies

Link every Source-1 business record to its matching records in Sources 2 and 3,
scored by macro F_0.5 over Source-1 entities.

Four stages, each a module with an explicit input and output on disk, each
runnable on its own:

| stage | module | reads | writes |
| --- | --- | --- | --- |
| A. normalize | `src.main --normalize-only` | the three source TSVs | `work/normalized/{split}_source{1,2,3}.parquet` |
| B. block | `src.blocking` | those parquets | `work/candidates/{split}_candidates.parquet` |
| C. featurize | `src.features` | candidates + normalized | `work/features/{split}_features.parquet` |
| D. score | `src.model` | the feature table | `output/matching_results.tsv`, `output/candidate_pairs.tsv` |

`src.main` runs all four end to end. `src.block_eval` scores stage B against the
training ground truth and is the diagnostic the blocking work was driven by.

## Setup

```bash
pip install -r requirements.txt
```

The data is not in this repo (`student_resource/` is gitignored, >1 GB). Unzip
it at the repo root so `student_resource/dataset/train/train_source1.tsv`
exists, or pass `--data-root`. Intermediates go to `work/`, overridable with
`--work-root` or `AMLC_WORK`.

Python 3.11+. Dependencies are polars, pyarrow, lightgbm, numpy,
indic-transliteration and psutil — all pip-installable with Windows wheels, no
build step, no GPU required.

## Reproducing the submission

Run these in order. Every stage caches its output, so a stage can be re-run
without repeating the ones before it. **`--split test` must run last**, because
both splits write to the same `--out-dir` and a later training run overwrites
the submission.

```bash
cd code/business_entity_resolution

# ---- train: normalize, block, featurize, fit the 5-fold ensemble ----
python -m src.main      --split train --normalize-only --translit --progress
python -m src.blocking  --split train --budget 60 --token-cap 500 --progress
python -m src.features  --split train --progress

# read MACRO RECALL off this and pass it as --coverage below
python -m src.block_eval --split train --budget 60 --token-cap 500 \
    --no-candidate-recall --no-miss-report --no-key-diagnostics \
    --budget-curve "60" --progress

python -m src.model     --split train --folds 5 --coverage <MACRO_RECALL> --progress

# ---- test: same blocking, the saved boosters, the submission ----
python -m src.main      --split test --normalize-only --translit --progress
python -m src.blocking  --split test --budget 60 --token-cap 500 --progress
python -m src.features  --split test --progress
python -m src.model     --split test --folds 5 --coverage <MACRO_RECALL> --progress
```

Outputs land in `output/matching_results.tsv` and `output/candidate_pairs.tsv`
at the repo root. Validate before submitting, from `student_resource/`:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

### Three flags that change the answer if omitted

* **`--translit`** must be passed to *both* `--normalize-only` calls. It
  romanizes Indic script before normalizing. Without it, cross-script true
  pairs are unreachable: 59.6% of the pairs no key could reach were
  Devanagari/Tamil/Kannada/Malayalam names against their Latin counterparts.
* **`--token-cap`** must match between `src.blocking` and `src.block_eval`, or
  the eval describes a run that did not happen. Same for `--band-cap`,
  `--acronym-cap`, `--acronym-min-len`, `--band-min-hits` and `--keys`.
* **`--coverage`** defaults to `1.0`, which makes the per-entity selection rule
  over-estimate recall and predict too few matches. Feed it the macro recall
  `src.block_eval` measures.

### Faster development loop

`--max-folds 1` trains one fold instead of five: an 80:20 grouped split that
saves one booster. Training drops from ~1h15m to ~16m and the test-split
ensemble from ~2h50m to ~35m, because prediction cost is linear in the number
of boosters (measured 61,500 rows/s per booster).

```bash
python -m src.model --split train --folds 5 --max-folds 1 --coverage C --progress
python -m src.model --split test  --folds 5 --coverage C --progress
```

The **train** submission from a partial run is not valid — only the held-out
fold's entities get a prediction — and `macro_f05` restricts its report to them
so the printed score stays honest. The **test** submission is complete either
way: every test row is scored by whatever boosters are saved. A single booster
scores slightly below the five-fold ensemble.

A smoke test that exercises every stage in about a minute:

```bash
python -m src.main --split train --sample 200000
```

`--sample` truncates Source 1 only and matches it against the **full** target
set, so the score stays representative. `--sample-targets` truncates the target
side too; it exists for memory pressure, warns, and makes the score meaningless,
because an entity's true partners are scattered across all 10M target rows.

### Seeing where a run is: `--progress`

A full run is hours long and, without this, each stage printed one line *after*
it finished — so a run about to die at minute 90 looked identical for 89
minutes to one that would succeed. `--progress` is on every entry point and
writes live lines to **stderr**: stage in/out with elapsed time, and a
rewritten-in-place position line per loop with rate, ETA, resident memory, peak
memory, free system memory and bytes written. Results go to stdout, so
`--progress > run.log` splits them.

Independently of the flag, every stage appends a JSON record to
`work/timings.jsonl` (disable with `--no-timings`), including a `stage_start`
record, so a killed run still says how far it got.

```bash
python -m src.timings              # last run, slowest stage first, with self time
python -m src.timings --tree       # nesting and execution order
python -m src.timings --runs 3     # compare the last three
```

Rank by the **self** column, not total — a parent's total includes its
children, so total always puts `model` on top and says nothing.

## How it works

### Stage A — normalize (`text_norm.py`, `translit.py`, `minhash.py`)

Raw name and address to every column the later stages block and score on.

Ordering is load-bearing. Romanization runs first, because `ACCENT_MAP` folds
the IAST diacritics it emits. Then: lowercase, fold accents, `&` → `and`, drop
apostrophes, fold punctuation to spaces, and glue runs of single letters into
one token so `S.B.I. ATM` does not become `south b i atm` — that gluing must
precede abbreviation expansion, or a lone `s` expands to `south`. Then
whole-token abbreviation expansion (one tokenize/map/join pass, not one regex
pass per entry — ~95 entries at 25s per million rows each would be five
minutes of every run), then legal-form and web-token stripping to give
`name_core`.

Accent handling strips only the Latin combining-diacritics block, never
`\p{M}`. In Indic scripts the vowel signs and virama are themselves
Mark-category characters, so a blanket strip deletes the vowels:
`डेवलपर्स` → `डवलपरस`, 33 characters down to 22.

Injected reference markers — `(ID: 84887)`, a trailing `#26282` — are stripped
from the raw name **before** punctuation folding, because afterwards there is
nothing left to match on but a bare number and bare numbers carry signal here.
They otherwise survive as ordinary tokens and make every name key disagree on a
pair whose names are otherwise identical. `--no-strip-noise` reproduces the
previous normalization.

Three passes, each over the materialized result of the last, not composed as
one expression: `name_nospace` → character 3-grams → 16 MinHash band
signatures. Built on top of the expression instead of its result, polars
recomputes the whole `name_core` chain once per slice — measured 246.7s per
million rows against 24.8s, i.e. 52 minutes of normalization instead of 5.

The result is cached and reused until the source data, the rules in
`text_norm.py`/`main.py`/`blocking.py`, the row count, the polars version, or
the `--translit` / `--no-strip-noise` settings change. The polars version is
part of the key because `name_trigrams` holds polars hashes, and hashes from
two builds are not comparable — without that guard an upgrade between runs
silently makes every cross-source trigram comparison return zero.

### Stage B — block (`blocking.py`)

Twenty-four keys by default, each paired with country, each with its own weight
and bucket cap. Country is folded into every key rather than compared
separately: it agrees on 100% of true training pairs, so it is a free hard
partition, and it is treated as an opaque label and never enumerated because
the test set contains France and the training set does not.

| key | kind | weight | default cap | what it catches |
| --- | --- | --- | --- | --- |
| `name_sorted` | exact | 1.00 | 2,000 | word-order transposition |
| `name_nospace` | exact | 0.80 | 2,000 | spacing differences |
| `addr_sorted` | exact | 0.85 | 2,000 | same address, different name |
| `name_acronym` | exact | 0.25 | 50 | initialism against expansion |
| `name_prefix` | exact | 0.75 | 2,000 | truncation, name concatenated into a domain |
| `band_00…15` | LSH | 0.10 | 100 | typos, character-level corruption |
| `name_tokens` | token | 0.30 | 40 | one shared rare word in the name |
| `addr_tokens` | token | 0.30 | 40 | one shared rare word in the address |
| `addr_nums` | token | 0.15 | 40 | shared house or postal number |

`name_core` exists but is off by default: it measured 0.00% unique contribution
twice.

**LSH bands** turn a similarity question into an equality one. MinHash reduces
each entity's 3-gram set to 48 integers whose agreement probability equals the
sets' Jaccard similarity; grouped into 16 bands of 3, a whole band agreeing is
an ordinary exact key, so the machinery above handles it unchanged. `P_MOD` is
`2**31-1` rather than a larger prime because **polars wraps UInt64 silently** —
at `2**61-1` a permutation overflows and yields a hash family that is neither
the intended one nor provably anything, with no exception raised. The bands are
nested, so a curve over `--bands` reads off one normalization pass.

**Token keys** are multi-valued: the normalized column is a list, the record
joins one bucket per token, and a pair is reachable if it shares *one* rare
token anywhere in either string. `_target_table` and `_s1_for_key` explode
list-typed keys and nothing else changes — an exploded row is
indistinguishable downstream from a single-valued key's row.

There is deliberately **no IDF filter** on the token keys. A token's bucket
size *is* its document frequency, so the bucket cap every key already has is
exactly an IDF cut, computed against the real corpus at block time rather than
guessed at in a stop-word list someone has to maintain: `private` lands in a
bucket of millions and is dropped by the same rule that keeps
`vaishnavisignature` in a bucket of two.

A candidate's `block_score` is the sum of the weights of the keys that found
it, each **divided by `log2(1 + bucket_size)`**. That inverse-frequency term is
what makes the score discriminative — sharing a name with one other record is
strong evidence, sharing it with 2,000 is nearly none. Without it, tied
candidates are evicted by the budget in effectively alphabetical order, worth
2.4 points of macro recall at an identical candidate count.

Scale is handled by looping key-outer rather than holding every key table
resident: build one target table, join all of Source 1 against it, spill the
hits to parquet partitioned by merge bucket, free it, move on. The merge then
processes one hash partition of the entity ID at a time — the budget is per
entity and entities are independent, so this is exact rather than approximate.
`projected_pairs` computes each key's exact emitted-pair count *before* writing
a row, and `--max-pairs-per-key` / `--max-pairs-total` refuse a configuration
that would not finish.

### Stage C — featurize (`features.py`)

56 features per candidate pair:

* **provenance** — `block_score`, `n_keys`, a `hit_<key>` flag per key unpacked
  from `key_mask`, `n_band_hits`, and `addr_only` (found by the address key and
  nothing else, which is exactly the circular case an address scorer cannot
  reject on its own).
* **name evidence** — trigram and token Jaccard *and* containment, exact-match
  flags on three normalizations, length and token-count ratios. Jaccard and
  containment disagree in the case that matters: `acme` against `acme foods
  private limited` has a low Jaccard and a containment of 1.0.
* **address evidence** — token Jaccard and containment, exact match, postal-code
  and house-number overlap, token counts, blank flags.
* **relative to the entity's own candidate list** — `n_cand`, and the fraction
  and rank of `block_score`, name trigram similarity and address overlap within
  that entity. These are what let the model say "this is the best of a weak
  field" versus "this is one of six equally good ones".

Set sizes are compared as `|A| + |B| - |A∩B|`, which is exact for sets and
wrong for multisets, so every list column reaching here is deduplicated at
build time.

### Stage D — score (`model.py`)

LightGBM binary classifier, `average_precision` metric, 5-fold **grouped** CV
over Source-1 entities: every entity is scored by the one fold model that did
not train on it, so the reported F_0.5 has no in-sample component. The fold
assignment is written into the feature table as a column and cross-checked
against `fold_of()` — if those ever disagreed, a fold would be scored by a
model that trained on it and the holdout score would be silently *inflated*,
which is the one failure mode that produces a better number rather than a
crash.

Selection is not a global threshold. For each entity the rule walks its
candidates in descending score and takes the prefix that maximizes expected
F_0.5 under the model's probabilities, de-biased by `--coverage` for the true
matches blocking never retrieved. `--topk` caps it at 8; the maximum true match
count is 11 and the 99th percentile is 8.

On the test split the boosters are loaded rather than trained, and every row is
scored by every booster. That is linear in ensemble size — 61,500 rows/s per
booster, measured — and is the single largest cost in a full submission run.

## Evaluating the blocking stage: `src.block_eval`

Blocking sets a hard ceiling on everything downstream: a true pair that never
becomes a candidate cannot be recovered by any model or threshold. This module
is how that ceiling is measured, and it has four sections.

```bash
# the ceiling, with no blocking run at all — reads the normalized parquet
python -m src.block_eval --split train --no-candidate-recall \
    --budget-curve="" --no-miss-report --cap-sweep "40,100,250,500,1000"

# what a candidate file actually delivered, plus 20 missed pairs in full
python -m src.block_eval --split train --budget 60 --token-cap 500 --show-misses 20
```

* **candidate-set recall** — pair recall, macro recall (the one that tracks the
  leaderboard, since F_0.5 is a macro average), entities fully covered,
  entities with nothing found, and the F_0.5 ceiling a perfect classifier could
  reach over that candidate set. The ceiling is computed *per entity* and then
  averaged: F is concave in recall, so plugging the mean recall into it
  overstates the mean F by about 9 points.
* **budget curve** — macro recall at every per-entity cut of the *same*
  candidate file. Blocking already ranked each entity's candidates, so a
  smaller budget is a `head(k)` on that ranking and the whole curve reads off
  one run.
* **missed-pair attribution** — every missed true pair assigned to the stage
  that lost it: `country-mismatch`, `no-key-agrees`, `bucket-cap`, `band-gate`,
  `budget-evicted`, or `unexplained`. That last one is a self-check, not a
  category: nothing in blocking drops a reachable pair for an unsaturated
  entity, so a non-zero count means the flags passed to the eval are not the
  ones that built the file. `--show-misses 20` then prints the
  `no-key-agrees` ones in full — raw and normalized name and address on both
  sides, which keys agreed, trigram Jaccard and containment — stratified across
  similarity bands so twenty rows span the range instead of landing in one
  corner of it.
* **token cap sweep** — reachability *and* exact emitted-pair cost at several
  token caps, from one pass and no blocking run. `UNION at cap` is a hard
  ceiling on pair recall: no budget can admit a pair no key generated. The cost
  columns are per key and flag the caps `--max-pairs-per-key` would refuse.
  Fan-out is `sum(source-1 rows in bucket × target rows in it)`, so it grows
  with the **square** of bucket size while recall grows logarithmically — the
  two columns have to be read together.

## Measured results

Full training split, 27 Sept 2026, with `--translit`.

### Reachability

The share of true pairs the key set can reach, before any budget:

| configuration | reachable at cap | reachable uncapped |
| --- | --- | --- |
| 5 exact keys only | 68.80% | — |
| + 16 LSH bands | 82.96% | 91.63% |
| + token keys, `name_prefix`, noise strip | 88.74% (cap 40) | **99.96%** |

Only **0.04%** of true pairs are now unreachable by any key at any cap. The
binding constraint moved from the key set to the bucket cap:

| token cap | reachable | raw pairs (3 token keys) |
| --- | --- | --- |
| 40 | 88.74% | 16.2M |
| 100 | 90.32% | 58.2M |
| 250 | 92.31% | 215.9M |
| **500** | **93.72%** | **496.1M** |
| 1,000 | 95.17% | 1.20B — refused |
| 2,000 | 96.52% | 3.45B — refused |

Cap 500 is the largest configuration that fits inside `--max-pairs-per-key`
(400M) with no key over the limit.

Per-key contribution at cap 40, over 7,638,365 true pairs:

| key | agrees | usable at cap | unique |
| --- | --- | --- | --- |
| `addr_tokens` | 95.19% | 20.20% | **3.33%** |
| `name_tokens` | 84.27% | 9.37% | 0.33% |
| `name_acronym` | 63.44% | 5.78% | 0.06% |
| `addr_nums` | 57.41% | 3.26% | 0.30% |
| `name_sorted` | 53.31% | 53.31% | 0.37% |
| `name_prefix` | 53.31% | 50.53% | **1.20%** |
| `name_nospace` | 53.20% | 53.20% | 0.08% |
| `addr_sorted` | 23.48% | 23.45% | **3.84%** |
| each band | ~61% | ~39% | ~0.12% |

Agreement and uniqueness are close to uncorrelated: `addr_sorted` has the
weakest agreement of the exact keys and the largest unique contribution, while
`name_core` agrees on half of all true pairs and contributes nothing.

### End to end

| | budget 20, 5 exact keys | budget 60, + 16 bands |
| --- | --- | --- |
| blocking macro recall | 59.98% | 76.09% |
| F_0.5 ceiling over those candidates | 0.7943 | 0.8794 |
| achieved out-of-fold | 0.7345 | **0.8121** |
| mean candidates / entity | 10.89 | 56.36 |

The model converts **92.4%** of the blocking ceiling into achieved score, and
that rate held to within 0.0012 across two runs with entirely different
candidate sets. Every point of reachability is therefore worth about 0.92
points of ceiling and 0.85 points of score.

## Knobs

| flag | stage | what it does |
| --- | --- | --- |
| `--budget` | B | candidates kept per entity, after ranking by `block_score` |
| `--token-cap` | B | bucket cap for the three token keys — the reachability knob |
| `--band-cap` | B | bucket cap for the 16 LSH bands — the fan-out knob |
| `--band-min-hits` | B | bands that must agree to admit a pair with no exact-key evidence |
| `--acronym-cap`, `--acronym-min-len` | B | `name_acronym`'s cap and minimum length |
| `--keys`, `--bands` | B | restrict the key set, for attributing a change to one key |
| `--max-pairs-per-key`, `--max-pairs-total` | B | refuse a configuration that would not finish |
| `--merge-buckets` | B, C | hash partitioning; affects peak memory only, never the output |
| `--folds`, `--max-folds` | D | grouped CV folds, and how many to actually train |
| `--coverage` | D | blocking's macro recall, de-biasing the selection rule |
| `--topk` | D | cap on predictions per entity |
| `--train-entities`, `--train-rows` | D | per-fold training-set size |

`--budget` and `--topk` are not the same kind of number. `--topk` is bounded by
the true match distribution. The budget has to be large enough that true
matches are not crowded out by *false* candidates that outrank them, which is a
question about ranking quality — `src.block_eval`'s budget curve answers it off
one candidate file instead of a sweep.

## Working practice

One change at a time, measured against the run it replaces, with the previous
behaviour reachable behind a flag rather than deleted — `--legacy-norm`,
`--no-strip-noise`, `--bands 0` and `--keys` all exist for that reason. A
regression then has to be attributable rather than bisected.

Three habits the code enforces, each after a failure that cost a run:

* **Every flag that changes what blocking emits must be repeated to the eval.**
  Recomputed from the normalized parquet, `block_eval` has no way to know which
  caps produced a candidate file and will describe the defaults unless told.
* **Any cached artefact records what produced it.** The normalized parquet
  stamps the polars version, `--translit` and `--no-strip-noise`; the model
  manifest records folds run and OOF completeness; stale blocking partials are
  deleted before new ones are written.
* **Cost is projected before it is paid.** `projected_pairs` is exact and runs
  before a row is emitted, because the alternative is finding out forty minutes
  in.
