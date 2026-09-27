# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Pyroflies  
**Team Members:** Poran Dip Boruah, Dikshayn Chakraborty, Rajdeep Choudhury, Ayan Nawchoo  
**Submission Date:** 27-09-2026

---

## 1. Executive Summary

We resolve 2.2M Source-1 businesses against 10.3M Source-2/3 records with a
four-stage pipeline: rule-based normalization, a 24-key blocking union that
mixes exact keys, MinHash/LSH bands and multi-valued *token* keys, 56 pair
features, and a LightGBM classifier with a per-entity expected-F_0.5 selection
rule instead of a global threshold. The core innovation is treating blocking as
a measured, attributable stage rather than a heuristic: every missed true pair
is assigned to the mechanism that lost it, and that diagnostic is what drove
the key set from reaching 82.96% of true pairs to **99.96%** — the remaining
0.04% being genuinely unreachable by any character- or token-level method.

---

## 2. Methodology

### 2.1 Problem Analysis

We drove EDA from failures rather than from summary statistics: we built a
diagnostic (`src.block_eval`) that prints the true pairs blocking missed, side
by side, with their normalized forms and which keys agreed. Reading those is
what produced every finding below.

**Noise is adversarial and mechanical.** Injected reference markers
(`(ID: 84887)`, a trailing `#26282`), bracketed legal forms (`[LP]`), leading
junk (`>> `), doubled spaces, and case inversion. The markers survive naive
normalization as ordinary tokens and make every name key disagree on a pair
whose names are otherwise identical — four of the first twenty unreachable
pairs we inspected.

**Names arrive in four different shapes for the same business:** word-order
transposed (`Range, Meridith Unified Lozano Inc`), concatenated into a domain
(`assetbuildingfund.com` vs `Asset Building Fund Care`), truncated (`Empire
Ministries of` vs `Empire Ministries of Yale LLC`), and character-corrupted
(`GREENVILLE INERNATIH ALLIED INSTITUTE`).

**Cross-script pairs are the hardest class.** A large share of Indian records
appear in Devanagari, Tamil, Kannada or Malayalam against Latin counterparts.
Romanization helps but is not sufficient: it renders `private limited` as
`bhiraivedh limidhedh`, so character similarity between the two sides collapses
to near zero. Of the true pairs no key could reach, **59.6% sat at trigram
Jaccard < 0.2**, and they were dominated by this class.

**The address is the field that survives.** Place names are written in Latin
far more often than business names are, so a rare address token
(`bayammathota`, `vaishnavisignature`, `boag`, `arangath`) is frequently the
only surviving agreement on a cross-script pair. Addresses are otherwise
heavily abbreviated and reordered (`8080 River Road, South Whitley, IN` vs
`8080 River Rd, S Whitley, Indiana`), and often blank.

**Country agrees on 100% of true training pairs**, making it a free hard
partition. The test set contains France, which training does not, so we treat
it as an opaque label and never enumerate it.

**Structural facts that bound the score.** 5.58% of Source-1 entities are true
singletons and score 1.0 for an empty prediction. The maximum true match count
is 11 and the 99th percentile is 8. F_0.5 is a *macro* average over entities,
so an entity with one true match weighs the same as one with eight — macro and
pair recall move apart, and only macro recall tracks the leaderboard.

### 2.2 Solution Strategy

**Approach Type:** Blocking + learned pairwise classifier + per-entity
selection rule.

**Core Innovation:** Two things, both in the retrieval half.

*Multi-valued token keys where the bucket cap is the IDF filter.* Every token
of ≥4 characters in the name and address becomes a blocking value of its own,
so a pair is a candidate if it shares **one rare token anywhere**. We apply no
stop-word list and no IDF threshold, because a token's bucket size *is* its
document frequency: the bucket cap every key already carries is exactly an IDF
cut, computed against the real corpus at block time. `private` lands in a
bucket of millions and is dropped by the same rule that keeps
`vaishnavisignature` in a bucket of two. Implementing this required one change
— exploding list-typed key columns before bucketing — after which an exploded
row is indistinguishable downstream from an ordinary key's row, so the census,
cost projection, cap, weighting, partitioning and merge all work unchanged.

*Blocking as a measured stage with per-failure attribution.* `src.block_eval`
classifies every missed true pair by the mechanism that lost it
(`country-mismatch` / `no-key-agrees` / `bucket-cap` / `band-gate` /
`budget-evicted`), sub-diagnoses the unreachable ones by name similarity, and
prints them in full. It also reports reachability *and* exact emitted-pair cost
at several bucket caps from a single pass with no blocking run. That turns key
design from guesswork into reading a table, and it is why the key set went from
82.96% to 99.96% reachable in one iteration rather than several.

---

## 3. Candidate Generation (Blocking)

**Blocking keys used — 24, unioned, each paired with country:**

| key | kind | weight | cap | catches |
| --- | --- | --- | --- | --- |
| `name_sorted` | exact | 1.00 | 2,000 | word-order transposition |
| `name_nospace` | exact | 0.80 | 2,000 | spacing differences |
| `addr_sorted` | exact | 0.85 | 2,000 | same address, different name |
| `name_acronym` | exact | 0.25 | 50 | initialism vs expansion |
| `name_prefix` | exact | 0.75 | 2,000 | truncation, domain concatenation |
| `band_00…band_15` | MinHash/LSH | 0.10 | 100 | typos, character corruption |
| `name_tokens` | token (multi-valued) | 0.30 | 500 | one shared rare name word |
| `addr_tokens` | token (multi-valued) | 0.30 | 500 | one shared rare address word |
| `addr_nums` | token (multi-valued) | 0.15 | 500 | shared house/postal number |

The LSH bands convert a *similarity* question into an *equality* one: MinHash
reduces each entity's character-3-gram set to 48 integers whose agreement
probability equals the sets' Jaccard similarity, and a whole band of 3 agreeing
is then an ordinary exact key. At 16 bands this detects trigram J = 0.5 about
88% of the time, and the bands are nested so a curve over band count reads off
one normalization pass.

A candidate's `block_score` is the sum of the weights of the keys that found
it, each divided by `log2(1 + bucket_size)`. That inverse-frequency term is
worth 2.4 points of macro recall at an identical candidate count: without it,
tied candidates are evicted by the per-entity budget in effectively
alphabetical order.

**Candidate pairs generated:** `<FILL: total rows in output/candidate_pairs.tsv>`
at `--budget 60 --token-cap 500`. For reference, the previous configuration
(budget 60, 21 keys, token cap 40) produced 124,383,366 pairs at 56.36 per
entity, a reduction ratio of 0.99999454 against 22.77 trillion possible pairs.

**How we ensured true matches were not lost.**

We measured it rather than assumed it, on three separate axes, all reported by
`src.block_eval` against the training ground truth.

*Reachability — can any key ever find this pair?* The union of all keys with
caps removed reaches **99.96%** of the 7,638,365 true pairs; only 0.04% are
unreachable by any of them. This is the number that bounds everything, because
no budget or model can recover a pair that was never generated.

| configuration | reachable uncapped |
| --- | --- |
| 5 exact keys | 76.26% |
| + 16 LSH bands | 91.63% |
| + token keys, `name_prefix`, noise strip | **99.96%** |

*Cap cost — does the bucket cap throw it away?* Reported per key as the gap
between `agrees` and `usable`. The token keys' fan-out grows with the **square**
of bucket size while reachability grows logarithmically, so the cap is chosen
from a sweep that prints both columns together:

| token cap | reachable at cap | raw pairs emitted |
| --- | --- | --- |
| 40 | 88.74% | 16.2M |
| 250 | 92.31% | 215.9M |
| **500** | **93.72%** | **496.1M** |
| 1,000 | 95.17% | 1.20B (refused by the pair guard) |

*Budget cost — does the per-entity cut evict it?* The budget curve reports
macro recall at every cut of a single candidate file, so the trade-off between
recall and candidate-set size is read off one run rather than swept.

Every missed pair that remains is attributed to one of those three mechanisms
plus `country-mismatch` and `band-gate`, with a sixth bucket, `unexplained`,
that exists purely as a self-check: nothing in blocking can drop a reachable
pair belonging to an unsaturated entity, so a non-zero count there means the
diagnostic's parameters do not match the run that produced the file, and it
says so rather than absorbing the pairs into a plausible-looking number.

---

## 4. Matching Model

**Features used — 56 per candidate pair.**

- **Name features:** character-trigram Jaccard **and** containment; token-set
  Jaccard and containment; exact-match flags on three normalizations
  (`name_sorted`, `name_nospace`, `name_acronym`); length ratio; token counts
  and their absolute difference. Jaccard and containment are both included
  because they disagree in exactly the case that matters here — `acme` against
  `acme foods private limited` has a low Jaccard and a containment of 1.0.
- **Address features:** token Jaccard and containment on `addr_core`; exact
  match; postal-code overlap (4–6 digit runs) and house-number overlap kept as
  *separate* features, since a shared postal code means same locality while a
  shared building number means same street address; token counts; blank flags
  for either side.
- **Provenance:** `block_score`, `n_keys`, a `hit_<key>` indicator per key
  unpacked from a bitmask, `n_band_hits`, and `addr_only` — found by the
  address key and nothing else, which is precisely the circular case an
  address-similarity score cannot reject on its own.
- **Relative to the entity's own candidate list:** `n_cand`, plus the fraction
  and rank of `block_score`, name trigram similarity and address overlap
  *within* that entity. These let the model distinguish "the best of a weak
  field" from "one of six equally good candidates", which a pairwise-only
  feature set cannot express and which matters directly under a macro metric.

**Model type:** LightGBM gradient-boosted trees, binary objective, average
precision as the early-stopping metric, 63 leaves, learning rate 0.05,
`min_data_in_leaf` 200, L2 1.0, up to 600 rounds with early stopping.

Trained with **5-fold grouped cross-validation over Source-1 entities**: every
entity is scored by the one fold model that did not train on it, so the
reported F_0.5 carries no in-sample component. The fold assignment is
materialized as a column in the feature table and cross-checked against the
function that computes it — if the two ever disagreed, a fold would be scored
by a model that had trained on it and the holdout score would be silently
*inflated*, which is the one failure mode that yields a better number instead
of a crash.

**Threshold selection method:** none — we do not use a global threshold.

A single cutoff is the wrong shape for a macro F_0.5 metric, because the
optimal number of predictions differs per entity: a confident singleton and an
entity with six plausible candidates want different answers. Instead, for each
entity the candidates are ranked by model probability and the rule takes the
**prefix that maximizes expected F_0.5** under those probabilities, capped at
`--topk 8` (the 99th percentile of true match counts is 8, the maximum is 11).

The expectation is de-biased by a `--coverage` parameter set to blocking's
measured macro recall, because the model's probabilities are conditioned on the
candidate set and cannot see the true matches blocking never retrieved; left at
1.0, the rule over-estimates recall and predicts too few.

---

## 5. Results & Error Analysis

**F_0.5 Score (macro):** `<FILL: out-of-fold macro F_0.5 from the final
src.model --split train run>`

Measured reference points on the full training split, all out-of-fold:

| configuration | blocking macro recall | ceiling over those candidates | achieved OOF F_0.5 |
| --- | --- | --- | --- |
| budget 20, 5 exact keys | 59.98% | 0.7943 | 0.7345 |
| budget 60, + 16 LSH bands | 76.09% | 0.8794 | **0.8121** |
| budget 60, + token keys (this submission) | `<FILL>` | `<FILL>` | `<FILL>` |

The model converts **92.4%** of the blocking ceiling into achieved score, and
that conversion rate held to within 0.0012 across two runs with entirely
different candidate sets. It is the most stable number we have, and it is what
makes the ceiling a usable planning tool: each point of reachability is worth
roughly 0.92 points of ceiling and 0.85 points of final score.

**Common false positives (wrong merges).** Three recurring shapes:

1. *Same building, different business.* Records sharing an address but not an
   entity — an office park, a registered-agent address, a shared PMB. These
   arrive through `addr_sorted` and now also `addr_nums`, and address
   similarity cannot reject them, which is exactly why `addr_only` and the
   per-key `hit_*` flags are features: the model learns that address evidence
   *alone* is weak.
2. *Same-family names distinguished by one token.* `Asset Building Fund Care`
   against `Asset Building Fund Central`, or `... Resort Private Limited`
   against `... Infratech Private Limited`. High trigram similarity, different
   entity. The relative-rank features help here by making "this scored 2.9 and
   the runner-up scored 2.8" visible.
3. *Generic high-frequency names.* `Generic Trading Company` style records
   where dozens of true distinct businesses share a normalized name. The
   inverse-frequency term in `block_score` demotes these, and the bucket cap
   removes the worst of them before scoring.

**Common false negatives (missed matches).** After the current key set, the
residual is small and concentrated. Of the pairs no key reaches at any cap
(0.04% of all true pairs), the breakdown by name similarity was:

- **59.6% at trigram J < 0.2** — cross-script pairs where romanization
  produces a phonetic rendering that shares almost no characters with the Latin
  form, *and* the address gives no rare shared token either. No character- or
  token-level method reaches these; they would need a multilingual embedding.
- **37.9% at J 0.2–0.5** — one heavily corrupted token with the rest intact.
  Mostly recovered now by `name_tokens`.
- **2.8% at trigram containment ≥ 0.9** — truncation and domain concatenation.
  Recovered by `name_prefix`.
- **0% at J ≥ 0.8** — the LSH bands already reach everything that similar, so
  none of the residual was a band-tuning problem.

The larger remaining loss is no longer reachability but the **bucket cap and
the per-entity budget**, which are cost-bounded rather than method-bounded: at
`--token-cap 500` the cap withholds pairs that a cap of 2,000 would reach, but
that configuration projects 3.45 billion raw pairs and does not fit the time
budget. A flat-zero entity — one where blocking found none of its true matches
— scores zero whatever the model does, so the share of such entities is the
hardest bound on macro F_0.5.

---

## 6. Conclusion

We treated candidate generation as a measurement problem rather than a
heuristic one: a diagnostic that attributes every missed true pair to the
mechanism that lost it took the reachable share of true pairs from 82.96% to
99.96% in a single iteration, mostly through multi-valued token keys in which
the existing bucket cap doubles as an IDF filter. The lesson we would carry
forward is that the ceiling is worth more attention than the model — with a
conversion rate that held at 92.4% across two very different candidate sets,
a point of recall was reliably worth 0.85 points of score, while the same
effort spent on the classifier was not. The honest limitation is that the
remaining gap is now paid for in compute rather than in ideas: the caps and
budget that would realize the last few points of reachability project billions
of candidate pairs, and choosing among them is a cost decision.

---

## Appendix

### A. Code Artefacts

Complete, runnable source ships under `code/business_entity_resolution/`, with
`README.md` (full reproduction steps, flag reference and design rationale) and
`requirements.txt`. Python 3.11+, pip-installable, Windows wheels for every
dependency, no GPU required.

```
code/business_entity_resolution/
├── README.md              reproduction steps, design notes, measured results
├── requirements.txt       polars, pyarrow, lightgbm, numpy,
│                          indic-transliteration, psutil
└── src/
    ├── main.py            pipeline entry point; all four stages end to end
    ├── text_norm.py       normalization rule tables and expression builders
    ├── translit.py        Indic → Latin romanization (--translit)
    ├── minhash.py         MinHash permutations and LSH band signatures
    ├── blocking.py        stage B: 24-key union, caps, scoring, merge
    ├── features.py        stage C: 56 pair features + labels
    ├── model.py           stage D: LightGBM grouped CV + selection rule
    ├── block_eval.py      blocking diagnostics (recall, misses, cap sweep)
    ├── paths.py           every file location in one place
    ├── progress.py        live progress + per-stage telemetry
    └── timings.py         reads work/timings.jsonl back
```

**Entry points to reproduce the submission.** Each stage caches its output, so
any stage can be re-run without repeating the ones before it. `--split test`
must run **last**: both splits write to the same output directory.

```bash
cd code/business_entity_resolution
pip install -r requirements.txt

# train
python -m src.main      --split train --normalize-only --translit
python -m src.blocking  --split train --budget 60 --token-cap 500
python -m src.features  --split train
python -m src.block_eval --split train --budget 60 --token-cap 500 \
    --no-candidate-recall --no-miss-report --no-key-diagnostics --budget-curve "60"
python -m src.model     --split train --folds 5 --coverage <MACRO_RECALL>

# test — writes output/matching_results.tsv and output/candidate_pairs.tsv
python -m src.main      --split test --normalize-only --translit
python -m src.blocking  --split test --budget 60 --token-cap 500
python -m src.features  --split test
python -m src.model     --split test --folds 5 --coverage <MACRO_RECALL>
```

`<MACRO_RECALL>` is printed by the `src.block_eval` line above. Add
`--progress` to any command for live stage/ETA/memory reporting on stderr;
every stage also appends timing and memory records to `work/timings.jsonl`,
readable with `python -m src.timings`.

A one-minute smoke test that exercises all four stages:
`python -m src.main --split train --sample 200000`.

### B. Additional Results

**Reachability by key group** (7,638,365 true training pairs, caps removed):

| group | reaches | adds uniquely |
| --- | --- | --- |
| exact keys alone | 83.18% | 18.52% |
| LSH bands alone | 70.22% | 5.56% |
| union of all 24 keys | **99.96%** | — |

**Per-key contribution** at token cap 40 (`agrees` = the key's recall with
unlimited buckets, `usable` = after the cap, `unique` = true pairs *no other
key* finds):

| key | agrees | usable | unique |
| --- | --- | --- | --- |
| `addr_tokens` | 95.19% | 20.20% | 3.33% |
| `name_tokens` | 84.27% | 9.37% | 0.33% |
| `name_acronym` | 63.44% | 5.78% | 0.06% |
| `addr_nums` | 57.41% | 3.26% | 0.30% |
| `name_sorted` | 53.31% | 53.31% | 0.37% |
| `name_prefix` | 53.31% | 50.53% | 1.20% |
| `name_nospace` | 53.20% | 53.20% | 0.08% |
| `addr_sorted` | 23.48% | 23.45% | 3.84% |
| `band_00…15` (each) | ~61% | ~39% | ~0.12% |

Agreement and uniqueness are close to uncorrelated, which is the argument for
keeping cheap keys that look redundant: `addr_sorted` has the weakest
agreement of the exact keys and the largest unique contribution, while
`name_core` agrees on half of all true pairs and contributes nothing, so it is
off by default.

**Macro recall against per-entity budget**, read off a single candidate file
(21 keys, token cap 40):

| budget | 1 | 3 | 8 | 20 | 30 | 60 |
| --- | --- | --- | --- | --- | --- | --- |
| macro recall | 21.42% | 45.90% | 62.52% | 70.18% | 72.46% | 76.34% |
| mean candidates/entity | 1.00 | 3.00 | 7.98 | 19.78 | 29.37 | 56.37 |

**Normalization ablation** (superseded single-key configuration, retained as
the attribution reference):

| normalization | macro F_0.5 | runtime |
| --- | --- | --- |
| legacy | 0.4670 | 27s |
| current rule set | 0.6070 | 176s |
| + romanization | 0.6075 | 302s |

Romanization looked worthless here — +0.0005 for +126s — because exact-key
blocking requires *every* token to agree and a partly-romanized name still
differs on its Latin tokens. Its real value is character-level: it moves
cross-script trigram similarity from 0.000 to 0.326, which only pays once
fuzzy and token retrieval exist. It is on for this submission.

**Cost model.** Blocking fan-out is `sum(source-1 rows in bucket × target rows
in bucket)`, projected exactly before any row is written. Test-split scoring is
linear in ensemble size at a measured 61,500 rows/s per booster, which is the
dominant cost of a full submission run and the reason the per-entity budget is
chosen against wall-clock as well as recall.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
