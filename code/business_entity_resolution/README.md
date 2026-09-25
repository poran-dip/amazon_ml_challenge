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

| column                    | what it is                                                                              | why                                                                      |
| ------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `name_norm`               | cleaned name, legal abbreviations expanded                                              | general-purpose                                                          |
| `name_core`               | `name_norm` minus legal-form tokens                                                     | `Acme Corp` == `Acme Corporation`                                        |
| `name_sorted`             | unique tokens, sorted                                                                   | word-order transpositions                                                |
| `name_nospace`            | `name_core` with spaces removed                                                         | `SunRise` == `Sun Rise`                                                  |
| `name_acronym`            | first letter of each token                                                              | links `IBM` to the expanded form                                         |
| `addr_norm` / `addr_core` | cleaned address, street abbreviations expanded, locality stopwords dropped from `_core` | `Rd` == `Road`                                                           |
| `addr_sorted`             | unique non-numeric tokens, sorted                                                       | component reordering                                                     |
| `addr_numbers`            | every digit run                                                                         | house/plot numbers are high-signal                                       |
| `addr_postal`             | digit runs of length 4–6                                                                | generic postal-code candidates                                           |
| `is_indic`                | record arrived in a non-Latin script                                                    | cross-script pairs are a different matching regime; useful model feature |

### Romanization

Training-set EDA found the same business appearing in _different scripts_ — `international systems` in one source, `ఇంటర్నేషనల్ ... సిస్టమ్స్` in another. No similarity measure can bridge that, so those true pairs are unreachable by any blocking strategy until both sides share an alphabet.

The Indic strings are not translations: `लिमिटेड` is the English word "limited" spelled phonetically. `src/translit.py` romanizes them (via IAST, whose diacritics the normal accent-folding path then flattens to ASCII), which lands within fuzzy-match distance of the Latin-script form. On the real backlog pairs, mean character-trigram similarity went from **0.000 to 0.326** — from impossible to reachable.

Romanized legal suffixes (`praivet` for "private") are in `LEGAL_TOKENS` alongside the Latin ones.

On fair play: this is a deterministic character mapping, the same class of operation as lowercasing — no external database, API, or data source is consulted, and nothing in it knows anything about any business. `indic-transliteration` is an MIT-licensed pure-Python library of script tables.

### 0. EDA (run this before touching the rules again)

```bash
python -m src.eda --sample 200000
```

Reports per-source profiles, blocking-key selectivity, and — the number that matters — **matched-pair agreement**: for each normalized field, the fraction of _true_ ground-truth pairs whose values are identical. That is the recall ceiling for any exact-match blocking key on that field, so it is the metric to move when changing a rule. It also prints true pairs that normalization failed to collapse, which is the rule backlog.

It also reports the match-count distribution per Source-1 entity and the singleton rate. That rate is the F_0.5 score of a submission that predicts nothing for everyone, so it is the floor any pipeline must beat. A per-source script census shows how much of each source arrived in a non-Latin script.

### 2. Blocking (candidate generation)

```bash
python -m src.blocking --split train
python -m src.block_eval --split train
```

Stage A is an exact-key union. No single key is enough — the best (`name_sorted`) agrees on only 52.8% of true pairs — but the keys are hash joins rather than similarity searches, and they disagree with each other, so their union recovers substantially more than the best of them alone. Keys used, weighted by how precise EDA showed them to be: `name_sorted`, `name_core`, `name_nospace`, `addr_sorted`, `name_acronym`. A candidate's score is the sum of the weights of the keys it matched on, so a record found by three keys outranks one found by an acronym collision alone.

Country is folded into every key as a hard partition. It agrees on 100% of true pairs, so it costs no recall and removes all cross-country comparisons.

Key values whose source-2/3 bucket exceeds `--max-bucket` (default 100) are dropped before the join. `name_acronym` has a 17,606-record bucket in source 1 alone; joining buckets that large emits hundreds of millions of pairs carrying almost no evidence.

`src/block_eval.py` reports pair recall (the hard ceiling on downstream recall), macro recall (per entity then averaged, which is how F_0.5 is scored), the share of entities whose matches were *all* found — the only ones that can still score 1.0 — and the count of entities left under budget, which is the workload handed to stage B.

`block_eval` also prints per-key diagnostics: for each key, how many true pairs it *agrees* on, how many survive the bucket cap (**usable**), what the cap therefore costs, and how many it finds that no other key finds (**unique**). A key with high agreement but no unique contribution is redundant; a key whose agreement is destroyed by the cap needs a higher `--max-bucket`, not a higher weight. The footer separates what stage A can reach from what is unreachable by any exact key at any cap — that residue is stage B's job and no rule change will touch it.

### Stage B — fuzzy n-gram search

```bash
python -m src.fuzzy --split train
python -m src.block_eval --split train --candidates <work>/candidates/train_candidates_final.parquet
```

Character 3-gram TF-IDF over `name_core`, reduced to 128 dense dimensions by truncated SVD (both fitted on a sample), L2-normalized so inner product is cosine, then one FAISS IVFPQ index per country. Top-k neighbours per Source-1 entity are filtered by `--min-sim`, merged with the stage A candidates, and the union is capped at the budget.

This is what recovers the failures that are character-level by construction: typos (`fnesnius`/`fresenius`), concatenation (`dentcoffee`/`coffee dent`), and transliteration residue (`kanstraksans`/`constructions`). All share most of their character n-grams while sharing no token exactly, so no exact key can ever reach them.

Embeddings are written to disk as memmaps and streamed into the index in chunks — holding 12.5M dense vectors in RAM would cost double-digit gigabytes. IVFPQ keeps only compressed codes resident, so a 6M-record country partition costs roughly 100MB rather than 3GB.

Stage A scores are key-weight sums up to ~3.5 while fuzzy similarities are 0–1, so `--fuzzy-weight` scales cosine to sit below a solid multi-key hit but above a lone acronym collision.

Later stages (features, training, prediction) land here as they are built.

## Design notes

**Country is an open set.** The test set contains France, which never appears in training. Nothing in the pipeline branches on a country value, enumerates `{US, India}`, or one-hot encodes country. Postal codes are extracted with a generic 4–6 digit-run rule rather than per-country regexes.

**Normalization is deliberately conservative.** Stage 1 only collapses variants that are _always_ equivalent (`Rd`/`Road`, `Pvt`/`Private`). Judgement calls — whether a landmark reference matters, how much a shared street name is worth — are left to the feature stage, where a model can weigh them, rather than having the rules destroy the information first.

**Vectorized, no Python UDFs.** All normalization is polars string expressions. At ~10M rows per split a per-row Python call would dominate runtime. Romanization is the one exception: the transliterator is per-string Python, so it runs behind a vectorized regex pre-filter that skips non-Indic rows entirely, and memoizes on unique input strings.

**Exact-match blocking is not viable on its own.** The best name key (`name_sorted`) agrees on 52.7% of true pairs; `name_acronym` reaches 60.7% but its largest bucket holds 17,606 records, so it cannot be a primary key. Blocking must be fuzzy — character n-gram similarity — partitioned by country, which agrees on 100% of true pairs and is therefore a free hard partition.

**Known gap:** region/state abbreviations (`IL` vs `Illinois`, and equivalents in other countries) are not expanded yet. EDA showed address is a weak signal overall (`addr_core` agrees on only 12% of true pairs, 62–65% of records have no postal code at all), so this is low priority — address belongs in the feature stage, not as a blocking key.

## Rule iteration loop

1. Edit a table in `src/text_norm.py` (they are plain dicts — the rules are data, not code).
2. `python -m src.normalize --split train --sample 200000`
3. `python -m src.eda --sample 200000`
4. Compare matched-pair agreement against the previous run; keep the change only if it went up.
