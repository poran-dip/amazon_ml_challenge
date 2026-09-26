# Entity Resolution Project — Full Development Checklist

**Re-audited against the code on 26 Sept 2026** (branch `feat/fuzzy-recall`, tip
`feat: python -m src.timings`). The previous version of this file said features
and the ML model were not started, when both exist and have scored; and it
claimed several things done that the code does not do. Every box below was
checked against a named file and symbol.

How to read the marks:

| | meaning |
| --- | --- |
| `[x]` | in the code, on the default path |
| `[~]` | partly done, or done but **stale / superseded** — the note says which |
| `[ ]` | not done |
| `[-]` | **deliberately not done.** The note says why. Not a gap to close. |

Where we actually are: **train out-of-fold macro F_0.5 = 0.7345, public
leaderboard 0.727.** The binding constraint is blocking recall (64% of true
pairs reachable; ~95% needed for the ~0.98 target), not the model.
`claude/findings.md` and `claude/next-steps.md` in the project hold the numbers
and the plan; this file is the coverage map.

Everything lives in `code/business_entity_resolution/src/`. `src/main.py` is the
only entry point that produces a submission.

---

## Phase 0 — Project Setup

- [x] Create project repository
- [x] Create project folder structure — but not the one Phase 19 describes; see
  the corrected tree there
- [x] Set up Python virtual environment
- [x] Create `requirements.txt` — polars, pyarrow, indic-transliteration,
  lightgbm, numpy, psutil
- [x] Install required libraries
- [x] Set up Git
- [x] Create `README.md`
- [x] Add dataset paths/configuration — `src/paths.py`, overridable by
  `--data-root` / `--work-root` / `--out-dir` or `AMLC_DATA` / `AMLC_WORK`
- [x] Create `.gitignore`
- [~] Create configuration file for thresholds/model parameters — **there is no
  config file.** Configuration is argparse defaults in each `main()` plus module
  constants: `blocking.KEY_SPECS` (weights and caps), `blocking.BAND_WEIGHT` /
  `BAND_CAP`, `minhash.DEFAULT_BANDS` / `DEFAULT_ROWS`,
  `model.DEFAULT_PARAMS`, `main.DEFAULT_MAX_BUCKET`. That is a defensible
  choice — the CLI is the config and it is recorded in `timings.jsonl`'s `argv`
  — but the box was checked for a file that does not exist.
- [~] Create separate directories for: — only `src/` exists in the repo.
  `work/` (normalized parquet, candidates, features, models, timings) and
  `output/` (the two TSVs) are created at run time and gitignored. There are
  **no** raw/processed/notebooks/experiments/config directories.
  - [x] Raw data — `student_resource/dataset/`, provided, gitignored
  - [x] Processed data — `work/normalized/`
  - [x] Candidate pairs — `work/candidates/`
  - [x] Features — `work/features/`
  - [x] Models — `work/model/` (boosters + `manifest.json`)
  - [x] Predictions — `output/`
  - [~] Evaluation — no directory; `src/block_eval.py` and `main.macro_f05`
    print to stdout. Per-stage cost goes to `work/timings.jsonl`.
  - [-] Notebooks — none, and none wanted. The pipeline is the artefact and it
    has to run unattended on two machines; a notebook is a second source of
    truth that cannot be diffed.
  - [x] Source code — `src/`

## Phase 0b — Observability *(new; added session 7)*

- [x] Per-stage timings, rows and peak memory to `work/timings.jsonl` —
  `src/progress.py`; `--progress` for live console output on stderr,
  `AMLC_PROGRESS=1` equivalently, `--no-timings` to suppress the file
- [x] Read the log back ranked by self time — `python -m src.timings`
- [x] Survive a killed run — a `stage_start` record is written on entry, so
  `src.timings` names the stages an interrupted run died inside
- [x] Disk accounting — bytes written by the process and net growth of
  `work/` + `output/`, per stage; blocking additionally reports its partial
  count and size
- [ ] Watchdog that kills a run past a time limit (`threading.Timer` +
  `os._exit`; `signal.alarm` is not on Windows)
- [ ] Per-stage time budgets from a real timed run — the numbers previously
  quoted for this were extrapolations under a candidate count that no longer
  applies

---

# Phase 1 — Dataset Understanding

## Load the data

- [x] Load Source 1 training data — `main.scan_source`
- [x] Load Source 2 training data
- [x] Load Source 3 training data
- [x] Load training ground truth — `block_eval.ground_truth_pairs`
- [x] Load test Source 1
- [x] Load test Source 2
- [x] Load test Source 3

Sources are read once and sliced, never re-scanned per chunk: `scan_csv(...)
.slice(offset, chunk).collect()` in a loop re-reads from the start every time.

## Inspect datasets

- [x] Check number of rows — ~12.5M per split; 2.2M source-1 entities,
  ~10.3M targets
- [x] Check number of columns
- [x] Check column names — `paths.SOURCE_COLUMNS`
- [x] Check data types — every column read as `Utf8`, deliberately: an ID that
  parses as a number in one file and not another is a silent join failure
- [x] Check missing values
- [x] Check duplicate records
- [x] Check duplicate IDs
- [x] Check unique countries — never enumerated in code; France is in test and
  not in train, so country is used as an opaque label
- [x] Check unique business names
- [x] Check address availability
- [x] Check record length distributions

## Understand ground truth

- [x] Determine how Source 1 IDs map to Source 2
- [x] Determine how Source 1 IDs map to Source 3
- [x] Identify entities with multiple matches — max 11, p99 = 8, which is what
  bounds `--topk`
- [x] Identify singleton Source 1 entities — 123,247
- [x] Determine exact ground-truth format — one row per source-1 ID, matches
  comma-joined in one column. A singleton's list is empty, and
  `ground_truth_pairs` filters on **length**, not nullness: polars 2.0 changes
  `empty_as_null` to False, at which point an `is_not_null` filter alone would
  admit a phantom empty-string match for all 123,247 singletons and make every
  recall number wrong.

---

# Phase 2 — Exploratory Data Analysis

- [x] Analyze business-name quality
- [x] Analyze address quality
- [x] Analyze country distribution
- [x] Analyze missing fields
- [x] Analyze duplicate names
- [x] Analyze duplicate addresses
- [x] Analyze name length
- [x] Analyze address length
- [x] Analyze true-match similarity
- [x] Analyze non-match similarity
- [x] Analyze singleton frequency
- [x] Identify common data-quality problems

## Save examples

- [x] Examples of exact matches
- [x] Examples of spelling variations
- [x] Examples of abbreviation variations
- [x] Examples of reordered names
- [x] Examples of address variations
- [x] Examples of difficult matches
- [x] Examples of false matches
- [ ] **Not in the repo.** The examples live in `claude/findings.md` and the
  error review in the project, not in a committed fixture. They should be a
  test file so a normalization change that breaks one of them fails a check.

---

# Phase 3 — Data Preprocessing

`src/text_norm.py` holds the rule tables; `main.prepare_full` applies them.
Rule ordering is load-bearing — initialism gluing must run before abbreviation
expansion.

## Business-name normalization

- [x] Convert to lowercase
- [x] Remove unnecessary punctuation
- [x] Normalize whitespace
- [x] Normalize special characters — strips **only** the Latin
  combining-diacritics block, never `\p{M}`. In Indic scripts the vowel signs
  and virama are Mark-category, so a blanket strip deletes the vowels.
- [x] Tokenize names
- [x] Handle word-order variations — `name_sorted`, an order-invariant token
  signature
- [x] Handle company suffixes — legal-form tokens stripped to give `name_core`
- [x] Handle abbreviations — whole-token lookup, not substring
- [x] Test normalization against real examples
- [x] Glue initialisms — `S.B.I. ATM` must not become `south b i atm`
- [~] Romanize Indic script — `src/translit.py`, behind `--translit`, **off by
  default**: +0.0005 F_0.5 for +126s measured. Its value is cross-script
  trigram similarity (0.000 → 0.326), which only pays now that the LSH bands
  consume trigrams. Turning it on is an open item.
- [ ] Transliteration variants (Shri / Shree / Sri and similar) — from the
  error review, not implemented
- [ ] French address abbreviations (`av`, `bd`, `pl`, `ch`, `imp`, `rte`) —
  France is ~8% of test and absent from train. Careful: in France `st` means
  Saint, not street.

## Address normalization

- [x] Convert to lowercase
- [x] Remove unnecessary punctuation
- [x] Normalize whitespace
- [x] Normalize address tokens — `addr_core`
- [x] Handle abbreviations — street-type lookup
- [x] Handle missing addresses — `fill_null("")`, and `addr_blank_e` /
  `addr_blank_c` carry the fact as features
- [x] Test normalization against real examples

## General

- [x] Preserve original values — normalized columns are added, never replaced
- [x] Create normalized columns separately
- [x] Check that normalization doesn't destroy useful information — the legacy
  rule folded to `[^a-z0-9]` and deleted every non-Latin character, putting
  ~1.5M target records out of reach of any key. `--legacy-norm` keeps that
  reachable as a measurement reference.
- [x] Cache normalization to parquet, invalidated by source mtime, row count,
  `--translit`, polars build **and the mtime of every rule module** —
  `main._is_fresh` / `_rule_mtime`. The guard refuses to import if a file in
  `src/` is classified in neither `_RULE_MODULES` nor `_NOT_RULE_MODULES`, so a
  new module cannot silently fall outside it.

---

# Phase 4 — Baseline Implementation

- [x] Run the existing baseline successfully
- [x] Verify data loading
- [x] Verify normalization
- [x] Verify blocking
- [x] Verify candidate generation
- [x] Verify similarity calculation
- [x] Verify thresholding
- [x] Verify top-K selection
- [x] Verify singleton handling
- [x] Generate baseline predictions
- [x] Generate baseline `candidate_pairs.tsv`
- [x] Generate baseline `matching_results.tsv`
- [x] Calculate baseline F₀.₅
- [x] Record baseline score — 0.4670, against a do-nothing floor of 0.0558

## Baseline experiment

```text
Baseline:
Blocking → Name
Scoring  → Address Jaccard
Decision → Fixed threshold
```

- [x] Save baseline results
- [x] Don't overwrite baseline when experimenting — `--legacy-norm` and
  `--blocking single` reproduce it in the current checkout, so a regression is
  attributable rather than bisected

---

# Phase 5 — Candidate Generation / Blocking

`src/blocking.py`. Five exact keys plus 16 LSH band keys = 21 keys
(`KEY_SPECS`), each with its own weight and bucket cap, unioned.

## Basic blocking

- [x] Implement country blocking — folded into **every** key as a hard
  partition rather than compared separately: it agrees on 100% of true training
  pairs, so it costs no recall and removes all cross-country comparisons
- [x] Implement normalized-name blocking — `name_core`, `name_nospace`
- [x] Implement name-token blocking — `name_sorted` (order-invariant)
- [x] Implement combined blocking — union of all keys, plus `addr_sorted` and
  `name_acronym`
- [x] Inverse-frequency weighting — a candidate's score is the sum of the
  weights of the keys that found it, **divided by `log2(1 + bucket_size)`**.
  Worth 2.4 points of macro recall at an *identical* candidate count: without
  it the per-entity budget evicts tied true matches effectively alphabetically.

## Advanced blocking

- [~] Character n-gram blocking — `blocking._emit_ngram` /
  `character_ngram_blocking` exist but `--char-ngram` **raises
  `NotImplementedError`**: it writes one unbucketed partial with no `key_bit`,
  which the current merge cannot consume, and it runs a Python lambda per row.
  Superseded by the LSH bands. Previously marked done.
- [~] Prefix blocking — same: `_emit_prefix` / `prefix_blocking` exist,
  `--prefix` raises `NotImplementedError`. Previously marked done.
- [~] TF-IDF retrieval — same: `_emit_tfidf` / `tfidf_retrieval_blocking`
  exist, `--tfidf` raises `NotImplementedError`, and as written it has no
  country split and densifies against all 10.3M targets. Previously marked
  done.
- [x] Multiple blocking keys
- [x] Union candidates from multiple blocking strategies
- [x] **MinHash / LSH banding** — `src/minhash.py`. 48 permutations over
  character 3-grams, 16 bands of 3 (`DEFAULT_BANDS=16`, `DEFAULT_ROWS=3`). Each
  band signature is an ordinary exact key, so the stage above handles it
  unchanged — same join, cap, weighting and budget, and **no new dependency**.
  23.74% of true pairs are unreachable by any exact key at any cap or budget,
  and the failures are character-level (`fnesnius`/`fresenius`,
  `dentcoffee`/`coffee dent`, `kanstraksans`/`constructions`).
  `P_MOD = 2**31-1`, not a larger prime, because **polars wraps UInt64
  silently**: at `2**61-1` a permutation overflows and yields a hash family
  that is provably nothing, with no exception raised.
- [x] `--bands N` blocks on the first N bands; `--bands 0` reproduces the
  exact-key set bit-for-bit. Bands are **nested**, so a curve over `--bands`
  reads off one normalization pass.
- [x] `--band-min-hits k` — k-of-b LSH. Only applies to pairs no exact key
  found, so `--bands 0` is provably unaffected. This is the candidate-set-size
  knob the scoring rule change calls for.
- [ ] Compare `rows=3` against `rows=4` — stricter, fewer candidates. Needs
  `--force-normalize`.
- [-] GPU sentence embeddings + FAISS — dropped on measurement: its
  truncated-SVD basis captured 34.3% of variance, so retrieval ran against a
  lossy projection of the similarity it was measuring, and it needed two
  dependencies with no GPU wheels on Windows.

## Blocking quality

- [x] Measure number of candidates — `block_eval.candidate_recall` reports
  mean candidates per entity, which is now a **scored** quantity
- [x] Measure candidate reduction
- [x] Check whether true matches survive blocking — macro and pair recall, and
  entities sitting **at** the budget cap with true matches still missing
- [x] Remove duplicate candidate pairs — the merge groups by
  (source1, candidate)
- [x] Remove excessively large blocks — per-key bucket caps; `--max-bucket`
  overrides them all with one value
- [x] Test blocking on difficult records
- [x] Guard against a run that would emit too much — `--max-pairs-per-key` and
  `--max-pairs-total`, both checked from an **exact** projection
  (`projected_pairs`) before a single row is written. The total matters
  independently: at `--band-cap 1000` each band projects ~212M, comfortably
  under any sane per-key limit, while sixteen come to 3.4 **billion**.
- [x] Delete stale partials before writing — a leftover from a run with a
  different key set, cap or `--merge-buckets` would otherwise be folded into
  this run's candidate set silently
- [x] Per-key cost recorded — pairs, bucket files and parquet MB per key in
  `work/timings.jsonl`
- [ ] Macro recall **per country** in `block_eval`

## Final candidate-generation system

- [x] Select blocking strategies
- [x] Implement candidate-generation pipeline — `blocking.block_split`
- [x] Generate final candidate pairs
- [x] Validate candidate coverage
- [x] Save `candidate_pairs.tsv` — `main.write_submission`
- [ ] **Fix the final configuration.** `--budget`, `--band-min-hits` and `rows`
  are not settled: the Pareto sweep (recall against mean candidates per entity,
  now that both are scored) has not been run on real data.

---

# Phase 6 — Similarity Feature Engineering

**Done.** `src/features.py` builds **52 feature columns** per candidate pair
(`FEATURE_COLUMNS`), bucketed by `hash(source1_entity_id) % --merge-buckets`.
The previous version of this file had every box here unchecked.

## Name features

- [x] Exact name match — `name_exact`, plus `name_nospace_exact` and
  `name_acronym_exact`
- [x] Name Jaccard similarity — `name_tok_jac`
- [x] Character n-gram similarity — `name_tri_jac`, `name_tri_cont`
  (3-gram Jaccard and containment)
- [x] Token overlap — `name_tok_cont` (containment, the asymmetric form)
- [x] Shared-token count — `name_ntok_e`, `name_ntok_c`, `name_ntok_absdiff`
- [x] Name length difference — `name_len_ratio`
- [-] Name Levenshtein similarity / edit distance — not implemented and not
  planned. Edit distance is O(nm) per pair in Python over ~124M pairs; the
  trigram Jaccard above measures the same character-level axis as a set
  operation polars vectorizes, and it is the quantity the LSH bands already
  estimate (corr 0.90).
- [-] Name TF-IDF cosine similarity — not implemented. The 3-gram vocabulary is
  15,285 distinct grams over ~258M target postings, so mean document frequency
  is ~17,000 and there are no rare grams for IDF to weight. The
  inverse-frequency term on `block_score` carries the rarity signal instead.

## Address features

- [x] Exact address match — `addr_exact`
- [x] Address Jaccard similarity — `addr_tok_jac`
- [x] Token overlap — `addr_tok_cont`
- [x] Shared-token count — `addr_ntok_e`, `addr_ntok_c`
- [x] Postal / number agreement — `addr_postal_overlap`,
  `addr_number_overlap`. The strongest non-name evidence available.
- [x] Missing-address indicator — `addr_blank_e`, `addr_blank_c`
- [-] Address Levenshtein / edit distance / TF-IDF cosine — same reasons as the
  name features above
- [ ] Address length difference — there is no `addr_len_ratio` to match
  `name_len_ratio`. Cheap; probably worth adding.
- [ ] Address character n-gram similarity — trigrams are computed for names
  only. The bands run on names, so the address side has no fuzzy feature at
  all, which is a real gap for `kanstraksans`-class address noise.

## Other features

- [-] Country exact match / country mismatch — **impossible by construction,
  and correctly so.** Country is a hard partition in every blocking key, so
  every candidate pair agrees on country and the feature would be the constant
  1. A tree splitting on a constant learns nothing.
- [ ] Missing-name indicator — no `name_blank_e` / `name_blank_c` to mirror the
  address pair. An empty `name_core` is currently only visible through
  `name_ntok_* == 0`.
- [x] Combined name/address similarity — the model combines them; no
  hand-built composite, which is the point of using a model
- [x] Provenance features — `block_score`, `n_keys`, `addr_only`,
  `n_band_hits`, and a `hit_<key>` flag for each of the 21 keys.
  `n_band_hits` is given to the model directly because the band count estimates
  trigram similarity (corr 0.90) and a tree needs sixteen splits to reconstruct
  a sum it can read in one.
- [x] **Relative-to-the-candidate-list features** — `n_cand`,
  `block_score_frac/rank`, `name_tri_frac/rank`, `addr_tok_frac/rank`. These
  are why sampling must be over **entities, never rows**: a partly sampled
  candidate list describes a list that never existed.

## Feature dataset

- [x] Generate feature matrix — `work/features/{split}_features.parquet`
- [x] Check for missing feature values — inner joins on both sides, and the
  featurized count is asserted equal to the candidate count. Carrying a pair
  with null features would let LightGBM split on "this row is broken".
- [x] Check feature distributions
- [x] Remove/handle invalid values
- [x] Save feature dataset
- [ ] Record feature importances **on real data** as the circularity check. On
  the fixture the address group still takes ~60% of total gain; if `addr_tok_*`
  dominates on real data the model has reproduced the old address scorer with
  extra steps.

---

# Phase 7 — Training Data Creation

**Done**, inside `features.build_features` rather than as a separate stage.

- [x] Generate positive candidate pairs — left join against the ground truth
- [x] Generate negative candidate pairs — every unlabelled candidate
- [x] Match candidates against ground truth
- [x] Assign labels:
  - [x] `1 = match`
  - [x] `0 = non-match` (`fill_null(0)`)
- [x] Check class balance — printed per run as positive / negative counts and
  the positive rate
- [x] Identify hard negatives — no separate mining step, and none needed: the
  negatives *are* hard, because every one of them shares a blocking key with
  the entity
- [x] Verify positive examples
- [x] Verify negative examples
- [x] Remove duplicate training pairs — the candidate file is unique on
  (source1, candidate) by construction
- [x] Refuse to train on a table with no `label` column rather than inventing
  zeros

## Final training structure

Actual, per row: `source1_entity_id`, `candidate_entity_id`, 52 feature
columns, `label` (train only).

---

# Phase 8 — Train Matching Models

`src/model.py`. **LightGBM only.**

## Model 1 — Baseline ML

- [-] Train Logistic Regression — not done and not planned. The baseline this
  project actually measures against is the rule-based address-Jaccard scorer
  (0.6070), which is a stronger and more honest floor than a linear model on
  the same features; adding sklearn for a model nobody would ship costs a
  dependency and a day.
- [x] Evaluate
- [x] Record F₀.₅ — rule baseline 0.6070

## Model 2 — XGBoost

- [-] Train XGBoost — not done and not planned. Same feature set, same model
  class, materially the same result; the deadline is better spent on blocking
  recall, which is the binding constraint. Revisit only if the model is ever
  shown to be the bottleneck.
- [-] Tune basic parameters
- [-] Evaluate
- [-] Record F₀.₅

## Model 3 — LightGBM

- [x] Train LightGBM — `model.train_cv`, binary objective, metric
  `average_precision`, `DEFAULT_PARAMS` (lr 0.05, 63 leaves,
  `min_data_in_leaf` 200, `feature_fraction` 0.9, `bagging_fraction` 0.8,
  `lambda_l2` 1.0)
- [x] Evaluate — fold AP ~0.9901
- [x] Record F₀.₅ — **0.7345 out-of-fold**, 0.727 public leaderboard
- [x] Bound its memory — training is planned on the **entity** table (~2.2M
  rows) not the pair table (~124M), each fold loads only its own rows, and
  `--train-rows` caps the training set by a **row** budget measured from the
  actual file. A limit expressed in entities cannot survive a change to the
  candidate set: rows per entity went from 10.9 to 56.4 when the bands landed,
  so `--train-entities 500000` silently became a 28.2M-row request.
- [x] Save boosters + feature manifest — `work/model/`
- [ ] Run `--rounds 2000` — early stopping has never fired at 600 on any fold,
  so the ceiling is untested. Under `--progress` LightGBM now logs its
  validation metric every 50 rounds, so the curve is visible during the run.
- [ ] Hyperparameter sweep on a fast dev loop

## Compare

- [x] Compare models using identical validation data — rule scorer vs model,
  same candidate set, same folds
- [x] Compare precision
- [x] Compare recall
- [x] Compare F₀.₅
- [x] Analyze false positives
- [x] Analyze false negatives
- [x] Select final model based on validation results — LightGBM ensemble of the
  5 fold models. Probabilities are averaged, not raw scores, because the
  selection rule multiplies `1 - p` and that is only meaningful on the
  probability scale. No sixth refit: each fold model saw 4/5 of the entities and
  their errors are only partly shared.
- [-] Keep the model country-blind — no per-country thresholds, one-hot
  encoding or `{US, India}` branches. France is in test and not in train, so
  anything country-specific cannot generalize to 8% of the test set.

---

# Phase 9 — Validation Strategy

- [x] Create training/validation split — 5-fold **grouped** CV, folds assigned
  by a hash of the source-1 entity ID (`model.fold_of`), so a run is
  reproducible without storing the split
- [x] Prevent validation leakage — grouping is by source-1 entity, so an
  entity's whole candidate list is in exactly one fold. Splitting on *pairs*
  would put an entity's candidates on both sides, and the relative features
  (`block_score_rank`, `n_cand`) describe the list, which is leakage by
  construction.
- [x] Separate inner split for early stopping — 1/20 of the training folds,
  independent hash seed, capped at a quarter of the training budget
- [x] Ensure validation contains realistic matching cases — every entity is
  scored by the one fold model that did not train on it
- [x] Evaluate positive matches
- [x] Evaluate non-matches
- [x] Evaluate singleton entities — scored, not excluded: an entity with no
  true matches scores 1.0 for an empty prediction and 0.0 for anything else
- [x] Evaluate difficult cases
- [x] Record validation results

## Metrics

- [x] Precision — macro, over entities that predicted something
- [x] Recall — macro, over entities with a true match
- [x] F₀.₅ — `main.macro_f05`, macro-averaged over **all** source-1 entities
- [x] False-positive count
- [x] False-negative count
- [x] Singleton accuracy/handling — true singletons counted separately, and
  `predicted empty` is split into entities that should be empty and entities
  that have real matches and score a flat zero
- [x] Blocking ceiling — `block_eval` reports the best F_0.5 reachable over a
  candidate set with a perfect classifier, which is what says whether to work
  on blocking or on the model. **Do not compute it as `F(mean p, mean r)`** —
  F is concave in r, so that overstates the macro mean by ~9 points on this
  data.
- [ ] Per-country breakdown of `macro_f05`
- [ ] Reliability table for `p` — the selection rule treats it as a real
  probability and its calibration has never been checked

---

# Phase 10 — Threshold Optimization

**Superseded on the model path.** There is no tuned threshold to optimize.

- [~] The sweep below was measured under the **superseded normalization** and
  no longer describes the pipeline. `addr_core` now expands abbreviations and
  drops stopwords, so the Jaccard distribution has moved.
  - [x] 0.15 (baseline default)
  - [x] 0.20 — F_0.5 = 0.5849
  - [x] 0.25 — F_0.5 = 0.5909
  - [x] 0.30 — F_0.5 = 0.5935 (peak)
  - [x] 0.35 — F_0.5 = 0.5932
- [-] 0.40+ for ML probability scoring — **not applicable.** `--scorer model`
  does not threshold. `model.select_matches` chooses, per entity, the prefix
  length k that maximizes an expected F_0.5:

      E[TP@k] = sum of top k probabilities
      P(k) = E[TP@k]/k,  R(k) = E[TP@k]/expected_true
      F(k) = 1.25 P R / (0.25 P + R)      -> take argmax_k

  and predicts nothing when `prod(1 - p)` beats the best `F(k)`. F_0.5 is
  macro-averaged **per entity**, so the question is never "is p > t" but "for
  this entity, does a fourth candidate help its own score" — which a global cut
  cannot express. It is a plug-in estimate, not a true expectation, and is
  documented as an approximation. **It has no tuned parameter**, which is the
  point: this project has twice agreed not to run that sweep.
- [x] `--coverage` de-biases the recall term, since the denominator is summed
  over the candidate set while blocking only reaches ~60% of true pairs.
  Measured three times at 1.0 / 0.93 / 0.75 → 0.8204 / 0.8205 / 0.8202.
  **It does nothing. That question is closed; do not spend a run on it.**
- [x] `--topk` caps k at 8 by default, from the p99 true match count
- [ ] Fix the under-prediction. On the fixture the rule predicts two thirds of
  the true match count (macro precision 0.963 against recall 0.655). If that
  reproduces on real data at AP 0.9901, `select_matches` is where the remaining
  points are — and `--coverage` is *not* the cause.

---

# Phase 11 — Match Ranking & Selection

- [x] Generate candidates — `blocking.block_split`
- [x] Generate features — `features.build_features`
- [x] Predict match probability — `model.predict_ensemble` /
  `predict_subset`, both streamed with `pyarrow.ParquetFile.iter_batches`. A
  `scan_parquet(...).slice(offset, chunk)` loop is **not** a cursor: each
  collect re-reads from the start, which at 23.6M rows is a dozen full passes.
- [x] Rank candidates — by model probability on the `--scorer model` path; by
  address Jaccard, then `block_score`, then `n_keys`, then candidate id on the
  `--scorer jaccard` path. Every tiebreak is explicit, because an unbroken tie
  selects survivors by row order.
- [x] Remove candidates below threshold — jaccard path only; the model path
  uses the selection rule above
- [x] Keep valid Source 2 matches
- [x] Keep valid Source 3 matches — sources 2 and 3 are one target pool
  throughout; nothing branches on which one a candidate came from
- [x] Handle multiple matches
- [x] Handle singleton entities
- [x] Remove duplicate links

## Final logic (actual)

```text
Source 1 (+ Sources 2/3)
   ↓ normalize            work/normalized/*.parquet   (cached)
   ↓ block: 5 exact keys + 16 LSH bands, per-entity budget
   ↓                      work/candidates/*.parquet   -> candidate_pairs.tsv
   ↓ features: 52 columns per pair
   ↓                      work/features/*.parquet
   ↓ LightGBM 5-fold ensemble -> p
   ↓ per-entity argmax_k expected F_0.5 (no threshold)
   ↓                      -> matching_results.tsv
```

---

# Phase 12 — Singleton Handling

- [x] Detect no-match entities — falls out of the selection rule rather than
  being a special case: the expected score of predicting nothing is
  `prod(1 - p)` over **all** the entity's candidates
- [x] Don't force weak matches
- [x] Verify true singleton cases — 123,247 in train
- [x] Test false-singleton cases
- [x] Every source-1 entity appears in the output exactly once, with an empty
  field if it has no matches — `main.write_submission`, from a **raw** scan of
  source 1, so an entity whose name normalized to nothing is still present. A
  missing entity is rejected outright by the scorer.
- [ ] Test entities with low-quality data
- [ ] Test entities with multiple possible candidates
- [ ] **173,911 entities get an empty prediction despite having real matches.**
  That is the largest single bucket of lost score and it is not yet attacked.

---

# Phase 13 — Error Analysis

- [x] Source 1 record
- [x] Candidate record
- [x] Predicted label
- [x] Actual label
- [x] Error category

## Classify errors

- [x] Spelling variation — now reachable via the LSH bands
- [x] Abbreviation
- [x] Word-order variation — `name_sorted`
- [x] Address variation
- [x] Transliteration / Indic script erasure — fixed in normalization (the
  Mark-category bug); `--translit` still off
- [x] Completely different name / DBA / synthetic alias — the irreducible
  residue; 2.97% of pairs are unreachable by any key on the fixture
- [x] Bucket cap dropping valid high-frequency tokens
- [x] Missing address
- [x] Missing name
- [x] Duplicate business names
- [x] Blocking failure — **the dominant cause.** 23.74% of true pairs are
  unreachable by any exact key; overall pair recall is 64%.
- [x] Threshold failure — reframed as selection-rule failure on the model path
- [ ] Turn the top three causes into owned tickets

---

# Phase 14 — Iterative Improvement

- [x] Identify problem
- [x] Modify preprocessing/blocking/features/model
- [x] Run validation
- [x] Calculate F₀.₅
- [x] Compare against previous version
- [x] Analyze whether improvement is genuine
- [x] Save experiment results — `claude/findings.md`; per-stage cost now also
  accumulates in `work/timings.jsonl`

## Experiment tracking

| Version | Blocking | Features | Model | Decision rule | F₀.₅ |
|---|---|---|---|---|---|
| V0 | legacy single key | token Jaccard | rules | threshold 0.15 | 0.4670 |
| V1 | exact multi-key | address Jaccard | rules | threshold 0.15 | 0.5781 |
| V2 | + char 5-gram + prefix | address Jaccard | rules | threshold 0.15 | 0.5765 |
| V3 | + char 5-gram + prefix | address Jaccard | rules | threshold 0.30 | 0.5935 |
| V4 | exact multi-key (current norm) | address Jaccard | rules | threshold 0.15 | 0.6070 |
| **V5** | **exact multi-key, budget 20** | **52 columns** | **LightGBM 5-fold** | **expected F_0.5** | **0.7345 OOF / 0.727 LB** |
| V6 | + 16 LSH bands | 52 columns | LightGBM 5-fold | expected F_0.5 | **not yet run on real data** |

V2 is a *negative* result kept deliberately: the extra keys cost 0.0016 at the
same threshold. So is `--acronym-min-len 4 --acronym-cap 500`, which measured
worse on **both** axes (63.73% recall for 46,995,103 candidates against 63.90%
for 46,671,107) — strictly dominated. Do not carry either forward.

Known ceilings, so effort goes where it can pay:

| candidate set | best reachable F_0.5 with a perfect classifier |
| --- | --- |
| current (budget 60, exact keys) | 0.8232 |
| capped union of exact keys | 0.8705 |
| uncapped union of exact keys | 0.9107 |
| 95% pair recall | ~0.985 |

**Exact-key blocking cannot exceed ~0.91 at any budget.** Against a ~0.98
target that settles where the work is.

---

# Phase 15 — Final Training

- [ ] Freeze preprocessing — `--translit` still undecided
- [ ] Freeze blocking strategy — `--budget` / `--band-min-hits` / `rows` still
  open pending the Pareto sweep
- [ ] Freeze feature set
- [ ] Freeze model
- [ ] Freeze hyperparameters — `--rounds` untested above 600
- [-] Freeze threshold — there is no threshold on the model path
- [ ] Retrain final model using available training data — decide between the
  5-fold ensemble and a refit on all entities; average 3 seeds if time allows
- [x] Save trained model — `save_models` writes boosters + `manifest.json`
  (features, fold count, params)
- [x] Save preprocessing configuration — stamped into the normalized parquet's
  schema metadata (polars version, `--translit`)
- [x] Save feature configuration — in the model manifest
- [x] Record the run's full configuration — `argv` in `work/timings.jsonl`

---

# Phase 16 — Test Prediction

`python -m src.main --split test --out-dir <its own dir>`. One code path, so
"identical" is structural rather than something to verify by eye.

- [x] Load test Source 1 / 2 / 3
- [x] Apply identical normalization — same `prepare_full`
- [x] Apply identical blocking — same `block_split`
- [x] Generate candidate pairs
- [x] Generate identical features — same `FEATURE_COLUMNS`; the manifest
  records them
- [x] Load final model — `load_models`
- [x] Predict probabilities — fold ensemble
- [x] Apply the selection rule
- [x] Rank matches
- [x] Handle multiple matches
- [x] Handle singletons
- [ ] **Actually run it end to end on test.** Not yet done. Two traps:
  **both splits default to the same `output/`**, so a training run afterwards
  silently overwrites the submission with training predictions — give test its
  own `--out-dir`; and check `--max-pairs-total` does not fire on test, which
  has a different country mix.

---

# Phase 17 — Required Output

## `candidate_pairs.tsv`

- [x] Generate final candidate pairs
- [x] Verify required columns — `source1_entity_id`, `candidate_entity_ids`
- [x] Verify IDs
- [x] Remove duplicate pairs
- [x] Ensure it represents candidates actually passed to matching — written
  from the candidate parquet, i.e. everything blocking reached, **not** what the
  model kept
- [x] **This file is now scored.** A smaller candidate set per source-1 entity
  ranks higher in the final evaluation, alongside the matching score. Current
  run: 21.15 candidates per entity. Recall and candidate count are two
  objectives, not one.

## `matching_results.tsv`

- [x] Generate final entity matches
- [x] Verify Source 1 IDs — every one present, exactly once
- [x] Verify Source 2/3 IDs
- [x] Verify singleton representation — empty field, not a missing row
- [x] Verify no duplicate matches
- [x] Verify formatting — TSV, `quote_style="never"`
- [x] Verify every Source 1 entity is accounted for
- [ ] Run `utils/validate_submission.py` on every candidate submission, from
  the `student_resource/` directory

---

# Phase 18 — Final Quality Checks

- [ ] Run pipeline from a clean environment
- [x] Confirm no hard-coded local paths — `src/paths.py` plus CLI and env
  overrides
- [x] Confirm no manual preprocessing — one entry point, no notebook steps
- [~] **Confirm reproducibility — this does NOT currently hold on the model
  path.** Measured 26 Sept: two identical `--scorer model` runs on identical
  inputs, same machine, produced **different** `matching_results.tsv`.
  `--scorer jaccard` was byte-identical. Likely cause is `DEFAULT_PARAMS`:
  `num_threads: 0` with `feature_fraction` / `bagging_fraction` and no
  `deterministic: true`, so multithreaded float accumulation moves scores
  enough to cross the selection rule on some entities. **Decide: pin
  determinism (`deterministic: true`, usually with `force_row_wise: true`,
  which costs speed and changes the score), or change the acceptance criterion
  to a score tolerance.** Do not leave it as "confirmed".
- [x] Confirm random seeds where appropriate — `--seed`, `FOLD_SEED`,
  `INNER_SEED`; folds derive from a hash of the entity ID, so the split is
  reproducible without being stored
- [x] Confirm model loads correctly — `load_models` + manifest
- [x] Confirm output files are generated
- [x] Confirm output format
- [x] Confirm no duplicate rows
- [x] Confirm no invalid IDs
- [x] Confirm singleton handling
- [x] Confirm candidate file matches actual pipeline
- [-] Confirm final threshold is documented — no threshold on the model path;
  the selection rule is documented in `model.select_matches`
- [x] Confirm peak memory stays under 16GB is now **checkable** —
  `python -m src.timings` reports per-stage peak RSS and the low-water mark of
  free system memory. Previously it was hoped for.
- [ ] **Pin the polars version.** Asked for in three consecutive handoffs and
  still unresolved. `name_trigrams` and the MinHash signatures are polars
  hashes, and `_is_fresh` refuses a cache written by a different build, so this
  is a correctness issue, not tidiness.
- [ ] Commit the 35 adversarial checks from session 5 under `tests/` and wire
  up pytest with `fast` / `scaling` / `full` markers plus `pytest-timeout`.
  **They are not in the repo**, so nothing currently stops a merge from
  breaking them.

---

# Phase 19 — Project Organization

**Actual structure.** The tree the previous version of this file recommended
(`data/`, `notebooks/`, `config/`, `experiments/`, one module per concern) was
never built and is not the plan — the pipeline is five stages in six modules,
and intermediates are addressed through `src/paths.py` rather than by
convention.

```text
<repo root>/
├── code/business_entity_resolution/
│   ├── src/
│   │   ├── main.py           entry point: normalize, block, score, predict, write, self-score
│   │   ├── paths.py          every path in one place; AMLC_DATA / AMLC_WORK / CLI overrides
│   │   ├── text_norm.py      normalization rule tables (edit data here, not code)
│   │   ├── translit.py       Indic romanization, behind --translit
│   │   ├── blocking.py       5 exact keys + 16 LSH band keys, budget, merge
│   │   ├── minhash.py        MinHash permutations and band signatures
│   │   ├── features.py       52 features per candidate pair
│   │   ├── model.py          LightGBM 5-fold CV, ensemble, selection rule
│   │   ├── block_eval.py     blocking diagnostics: recall, budget curve, per-key reach
│   │   ├── progress.py       run telemetry -> stderr + work/timings.jsonl
│   │   ├── timings.py        read work/timings.jsonl back, ranked by self time
│   │   └── test/             one experiment harness; NOT a test suite yet
│   ├── requirements.txt
│   ├── README.md
│   └── Entity Resolution Checklist.md   (this file)
│
├── student_resource/     provided data + utils/validate_submission.py  (gitignored)
├── work/                 normalized/ candidates/ features/ model/ timings.jsonl  (gitignored)
└── output/               candidate_pairs.tsv, matching_results.tsv  (gitignored)
```

Both splits write to `output/` by default, so `--split test` runs **last** or
gets its own `--out-dir`.

---

# Phase 20 — Documentation

`Documentation_template.md` at the repo root is **still blank**, and the
submission date on it says 27 Sept. `README.md` in this directory is current and
detailed and is the best source to fill it from.

- [x] Project overview — README
- [x] Problem definition — README
- [x] Dataset description — README (*Scale*)
- [x] Data preprocessing — README (*How it works*)
- [x] Blocking methodology — README (*Block*, *Fuzzy retrieval*)
- [ ] Feature engineering — not written up anywhere but the code
- [ ] ML model — not written up anywhere but the code
- [ ] Training methodology
- [ ] Validation methodology
- [x] Evaluation metric — README
- [-] Threshold selection — replaced by the selection rule; document that
  instead, and say why there is no threshold
- [x] Singleton handling — `model.select_matches` docstring
- [x] Error analysis — `claude/findings.md`
- [ ] Final results
- [ ] Limitations
- [x] How to run the project — README (*Run*)
- [x] Input/output format — README
- [~] Reproducibility instructions — blocked on the determinism finding in
  Phase 18 and the unpinned polars version
- [x] Observability — README (*Seeing where a run is*)

---

# Final Completion Checklist

## DATA

- [x] Data loaded
- [x] Data analyzed
- [x] Ground truth understood

## PREPROCESSING

- [x] Name normalization
- [x] Address normalization
- [x] Missing-value handling

## CANDIDATE GENERATION

- [x] Blocking implemented
- [x] Multiple blocking strategies tested
- [x] Candidate recall checked — 64% pair recall; ~95% needed
- [x] `candidate_pairs.tsv` generated
- [ ] Final blocking configuration fixed (Pareto sweep not run)

## FEATURES

- [x] Name features
- [x] Address features
- [-] Country features — constant by construction; see Phase 6
- [-] TF-IDF features — no rare grams to weight; see Phase 6
- [-] Edit similarity — trigram Jaccard covers the axis vectorized; see Phase 6
- [x] Jaccard similarity
- [x] Provenance and relative-to-list features

## MODEL

- [x] Baseline model — rule-based address Jaccard, 0.6070
- [-] XGBoost tested — deliberate non-goal
- [x] LightGBM tested — 0.7345 OOF
- [x] Final model selected — LightGBM 5-fold ensemble
- [ ] Final model retrained under the frozen configuration

## EVALUATION

- [x] Train/validation split — grouped 5-fold by entity
- [x] Precision
- [x] Recall
- [x] F₀.₅
- [-] Threshold tuning — no threshold on the model path
- [x] Error analysis
- [x] Blocking ceiling computed

## MATCHING

- [x] Match ranking
- [x] Multiple matches
- [x] Singleton detection
- [x] Duplicate prevention

## FINAL

- [ ] Final model trained
- [ ] Test prediction generated — **the pipeline has never been run on test**
- [x] `candidate_pairs.tsv`
- [x] `matching_results.tsv`
- [ ] Output validated — `validate_submission.py` not yet run on a real
  submission
- [~] Pipeline reproducible — jaccard path yes, model path **no** (Phase 18)
- [ ] Documentation complete

---

# 10 Major Milestones

- [x] **M1 — Data + EDA**
- [x] **M2 — Preprocessing**
- [x] **M3 — Baseline reproduced** — 0.4670 legacy, 0.6070 current rules
- [x] **M4 — Blocking improved** — 5 exact keys + 16 LSH bands; **not yet
  measured on real data with the bands on**
- [x] **M5 — Similarity features completed** — 52 columns. *Was marked
  unstarted; it has been done for two sessions.*
- [x] **M6 — ML training pipeline completed** — LightGBM 5-fold grouped CV,
  ensemble prediction, per-entity selection rule. *Was marked unstarted.*
- [~] **M7 — Threshold + F₀.₅ optimization** — the address-threshold sweep is
  done but stale; the model path replaced thresholding with the expected-F_0.5
  rule, whose **under-prediction is the open precision problem**
- [x] **M8 — Error analysis + improvements** — root cause identified as
  blocking recall, and the fuzzy stage built to attack it
- [ ] **M9 — Final test pipeline + outputs** — blocked on: the Pareto sweep,
  one test-split run, `validate_submission.py`, and the determinism decision
- [ ] **M10 — Documentation + submission** — `Documentation_template.md` blank

## The three things actually in the way

1. **Blocking recall: 64%, need ~95%.** Nothing else can buy the target. The
   bands are written and untested on real data.
2. **The selection rule under-predicts** — 173,911 entities get nothing despite
   having real matches. `--coverage` is not the cause; that is closed.
3. **The model path is not reproducible**, so the release check as written
   cannot pass.
