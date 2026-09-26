# Entity Resolution Project — Full Development Checklist

## Phase 0 — Project Setup

- [x] Create project repository
- [x] Create project folder structure
- [x] Set up Python virtual environment
- [x] Create `requirements.txt`
- [x] Install required libraries
- [x] Set up Git
- [x] Create `README.md`
- [x] Add dataset paths/configuration
- [x] Create `.gitignore`
- [x] Create configuration file for thresholds/model parameters
- [x] Create separate directories for:
  - [x] Raw data
  - [x] Processed data
  - [x] Candidate pairs
  - [x] Features
  - [x] Models
  - [x] Predictions
  - [x] Evaluation
  - [x] Notebooks
  - [x] Source code

---

# Phase 1 — Dataset Understanding

## Load the data

- [x] Load Source 1 training data
- [x] Load Source 2 training data
- [x] Load Source 3 training data
- [x] Load training ground truth
- [x] Load test Source 1
- [x] Load test Source 2
- [x] Load test Source 3

## Inspect datasets

- [x] Check number of rows
- [x] Check number of columns
- [x] Check column names
- [x] Check data types
- [x] Check missing values
- [x] Check duplicate records
- [x] Check duplicate IDs
- [x] Check unique countries
- [x] Check unique business names
- [x] Check address availability
- [x] Check record length distributions

## Understand ground truth

- [x] Determine how Source 1 IDs map to Source 2
- [x] Determine how Source 1 IDs map to Source 3
- [x] Identify entities with multiple matches
- [x] Identify singleton Source 1 entities
- [x] Determine exact ground-truth format

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

---

# Phase 3 — Data Preprocessing

## Business-name normalization

- [x] Convert to lowercase
- [x] Remove unnecessary punctuation
- [x] Normalize whitespace
- [x] Normalize special characters
- [x] Tokenize names
- [x] Handle word-order variations
- [x] Handle company suffixes
- [x] Handle abbreviations
- [x] Test normalization against real examples

## Address normalization

- [x] Convert to lowercase
- [x] Remove unnecessary punctuation
- [x] Normalize whitespace
- [x] Normalize address tokens
- [x] Handle abbreviations
- [x] Handle missing addresses
- [x] Test normalization against real examples

## General

- [x] Preserve original values
- [x] Create normalized columns separately
- [x] Check that normalization doesn't destroy useful information

---

# Phase 4 — Baseline Implementation

Use the supplied baseline as the starting point.

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
- [x] Record baseline score

## Baseline experiment

```text
Baseline:
Blocking → Name
Scoring  → Address Jaccard
Decision → Fixed threshold
```

- [x] Save baseline results
- [x] Don't overwrite baseline when experimenting

---

# Phase 5 — Candidate Generation / Blocking

## Basic blocking

- [x] Implement country blocking
- [x] Implement normalized-name blocking
- [x] Implement name-token blocking
- [x] Implement combined blocking

## Advanced blocking

- [x] Character n-gram blocking
- [x] Prefix blocking
- [x] TF-IDF retrieval
- [x] Multiple blocking keys
- [x] Union candidates from multiple blocking strategies

## Blocking quality

- [x] Measure number of candidates
- [x] Measure candidate reduction
- [x] Check whether true matches survive blocking
- [x] Remove duplicate candidate pairs
- [x] Remove excessively large blocks
- [x] Test blocking on difficult records

## Final candidate-generation system

- [x] Select blocking strategies
- [x] Implement candidate-generation pipeline
- [x] Generate final candidate pairs
- [x] Validate candidate coverage
- [x] Save `candidate_pairs.tsv`

---

# Phase 6 — Similarity Feature Engineering

Create features for every candidate pair.

## Name features

- [ ] Exact name match
- [ ] Name Jaccard similarity
- [ ] Name Levenshtein similarity
- [ ] Name edit distance
- [ ] Name TF-IDF cosine similarity
- [ ] Character n-gram similarity
- [ ] Token overlap
- [ ] Shared-token count
- [ ] Name length difference

## Address features

- [ ] Exact address match
- [ ] Address Jaccard similarity
- [ ] Address Levenshtein similarity
- [ ] Address edit distance
- [ ] Address TF-IDF cosine similarity
- [ ] Character n-gram similarity
- [ ] Token overlap
- [ ] Shared-token count
- [ ] Address length difference

## Other features

- [ ] Country exact match
- [ ] Country mismatch
- [ ] Missing-name indicator
- [ ] Missing-address indicator
- [ ] Combined name/address similarity
- [ ] Name-to-address consistency features

## Feature dataset

- [ ] Generate feature matrix
- [ ] Check for missing feature values
- [ ] Check feature distributions
- [ ] Remove/handle invalid values
- [ ] Save feature dataset

---

# Phase 7 — Training Data Creation

Convert candidate pairs into ML training examples.

- [ ] Generate positive candidate pairs
- [ ] Generate negative candidate pairs
- [ ] Match candidates against ground truth
- [ ] Assign labels:
  - [ ] `1 = match`
  - [ ] `0 = non-match`
- [ ] Check class balance
- [ ] Identify hard negatives
- [ ] Verify positive examples
- [ ] Verify negative examples
- [ ] Remove duplicate training pairs

## Final training structure

```text
source1_id
source2_id
name_jaccard
name_tfidf
address_jaccard
...
country_match
label
```

---

# Phase 8 — Train Matching Models

## Model 1 — Baseline ML

- [ ] Train Logistic Regression
- [ ] Evaluate
- [ ] Record F₀.₅

## Model 2 — XGBoost

- [ ] Train XGBoost
- [ ] Tune basic parameters
- [ ] Evaluate
- [ ] Record F₀.₅

## Model 3 — LightGBM

- [ ] Train LightGBM
- [ ] Tune basic parameters
- [ ] Evaluate
- [ ] Record F₀.₅

## Compare

- [ ] Compare models using identical validation data
- [ ] Compare precision
- [ ] Compare recall
- [ ] Compare F₀.₅
- [ ] Analyze false positives
- [ ] Analyze false negatives
- [ ] Select final model based on validation results

---

# Phase 9 — Validation Strategy

- [ ] Create training/validation split
- [ ] Prevent validation leakage
- [ ] Ensure validation contains realistic matching cases
- [ ] Evaluate positive matches
- [ ] Evaluate non-matches
- [ ] Evaluate singleton entities
- [ ] Evaluate difficult cases
- [ ] Record validation results

## Metrics

- [ ] Precision
- [ ] Recall
- [ ] F₀.₅
- [ ] False-positive count
- [ ] False-negative count
- [ ] Singleton accuracy/handling

---

# Phase 10 — Threshold Optimization

Don't assume `0.5`.

Test:

- [x] 0.15 (Baseline default)
- [x] 0.20 (Tuned on expanded candidates: F_0.5 = 0.5849)
- [x] 0.25 (Tuned on expanded candidates: F_0.5 = 0.5909)
- [x] 0.30 (Peak threshold on expanded candidates: F_0.5 = 0.5935)
- [x] 0.35 (Tuned on expanded candidates: F_0.5 = 0.5932)
- [ ] 0.40+ (For multi-feature ML probability scoring)

For each threshold:

- [x] Calculate precision
- [x] Calculate recall
- [x] Calculate F₀.₅
- [x] Count false merges
- [x] Count missed matches
- [x] Check singleton behavior

- [x] Select initial threshold based on validation performance (0.30 for multi-blocking baseline)
- [x] Save threshold in configuration

---

# Phase 11 — Match Ranking & Selection

For each Source 1 entity:

- [x] Generate candidates
- [ ] Generate features
- [ ] Predict match probability
- [x] Rank candidates (by address Jaccard, block score, and key count)
- [x] Remove candidates below threshold
- [x] Keep valid Source 2 matches
- [x] Keep valid Source 3 matches
- [x] Handle multiple matches
- [x] Handle singleton entities
- [x] Remove duplicate links

## Final logic

```text
Source 1
   ↓
Candidates
   ↓
ML probability
   ↓
Threshold
   ↓
Rank
   ↓
Valid matches
```

---

# Phase 12 — Singleton Handling

Explicitly implement:

- [x] Detect no-match entities
- [x] Don't force weak matches
- [x] Verify true singleton cases
- [x] Test false-singleton cases
- [ ] Test entities with low-quality data
- [ ] Test entities with multiple possible candidates

---

# Phase 13 — Error Analysis

Create an error-analysis dataset.

For every incorrect prediction:

- [x] Source 1 record
- [x] Candidate record
- [x] Predicted label
- [x] Actual label
- [x] Error category

## Classify errors

- [x] Spelling variation
- [x] Abbreviation
- [x] Word-order variation
- [x] Address variation
- [x] Transliteration / Indic script erasure
- [x] Completely different name / DBA / synthetic alias
- [x] Bucket cap dropping valid high-frequency tokens
- [x] Missing address
- [x] Missing name
- [x] Duplicate business names
- [x] Blocking failure
- [x] Threshold failure

---

# Phase 14 — Iterative Improvement

For each improvement:

- [x] Identify problem
- [x] Modify preprocessing/blocking/features/model
- [x] Run validation
- [x] Calculate F₀.₅
- [x] Compare against previous version
- [x] Analyze whether improvement is genuine
- [x] Save experiment results

## Experiment tracking

| Version | Blocking | Features | Model | Threshold | F₀.₅ |
|---|---|---|---|---|---|
| V0 | Baseline Legacy Single-Key | Token Jaccard | Rules | 0.15 | 0.4670 |
| V1 | Baseline Exact Multi-Key | Address Jaccard | Rules | 0.15 | 0.5781 |
| V2 | Exact Multi-Key + Char 5-gram + Prefix | Address Jaccard | Rules | 0.15 | 0.5765 |
| V3 | Exact Multi-Key + Char 5-gram + Prefix | Address Jaccard | Rules | 0.30 | 0.5935 |
| V4 | Exact Multi-Key + Sparse TF-IDF + N-gram | Address Jaccard + Multi-Features | ML (XGBoost/LightGBM) | Tuned | TBD |

---

# Phase 15 — Final Training

Once the architecture is finalized:

- [ ] Freeze preprocessing
- [ ] Freeze blocking strategy
- [ ] Freeze feature set
- [ ] Freeze model
- [ ] Freeze hyperparameters
- [ ] Freeze threshold
- [ ] Retrain final model using available training data
- [ ] Save trained model
- [ ] Save preprocessing configuration
- [ ] Save feature configuration
- [ ] Save threshold

---

# Phase 16 — Test Prediction

Run the final pipeline on the test data.

- [ ] Load test Source 1
- [ ] Load test Source 2
- [ ] Load test Source 3
- [ ] Apply identical normalization
- [ ] Apply identical blocking
- [ ] Generate candidate pairs
- [ ] Generate identical features
- [ ] Load final model
- [ ] Predict probabilities
- [ ] Apply final threshold
- [ ] Rank matches
- [ ] Handle multiple matches
- [ ] Handle singletons

---

# Phase 17 — Required Output

## `candidate_pairs.tsv`

- [ ] Generate final candidate pairs
- [ ] Verify required columns
- [ ] Verify IDs
- [ ] Remove duplicate pairs
- [ ] Ensure it represents candidates actually passed to matching

## `matching_results.tsv`

- [ ] Generate final entity matches
- [ ] Verify Source 1 IDs
- [ ] Verify Source 2/3 IDs
- [ ] Verify singleton representation
- [ ] Verify no duplicate matches
- [ ] Verify formatting
- [ ] Verify every Source 1 entity is accounted for

---

# Phase 18 — Final Quality Checks

- [ ] Run pipeline from a clean environment
- [ ] Confirm no hard-coded local paths
- [ ] Confirm no manual preprocessing
- [ ] Confirm reproducibility
- [ ] Confirm random seeds where appropriate
- [ ] Confirm model loads correctly
- [ ] Confirm output files are generated
- [ ] Confirm output format
- [ ] Confirm no duplicate rows
- [ ] Confirm no invalid IDs
- [ ] Confirm singleton handling
- [ ] Confirm candidate file matches actual pipeline
- [ ] Confirm final threshold is documented

---

# Phase 19 — Project Organization

Recommended structure:

```text
entity-resolution/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── candidates/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_features.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_error_analysis.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── blocking.py
│   ├── similarity.py
│   ├── features.py
│   ├── training.py
│   ├── prediction.py
│   ├── evaluation.py
│   └── pipeline.py
│
├── models/
│
├── outputs/
│   ├── candidate_pairs.tsv
│   └── matching_results.tsv
│
├── experiments/
│
├── config/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Phase 20 — Documentation

- [ ] Project overview
- [ ] Problem definition
- [ ] Dataset description
- [ ] Data preprocessing
- [ ] Blocking methodology
- [ ] Feature engineering
- [ ] ML model
- [ ] Training methodology
- [ ] Validation methodology
- [ ] Evaluation metric
- [ ] Threshold selection
- [ ] Singleton handling
- [ ] Error analysis
- [ ] Final results
- [ ] Limitations
- [ ] How to run the project
- [ ] Input/output format
- [ ] Reproducibility instructions

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
- [x] Candidate recall checked
- [x] `candidate_pairs.tsv` generated

## FEATURES

- [ ] Name features
- [ ] Address features
- [ ] Country features
- [ ] TF-IDF features
- [ ] Edit similarity
- [ ] Jaccard similarity

## MODEL

- [x] Baseline model
- [ ] XGBoost tested
- [ ] LightGBM tested
- [ ] Final model selected

## EVALUATION

- [x] Train/validation split
- [x] Precision
- [x] Recall
- [x] F₀.₅
- [x] Threshold tuning
- [x] Error analysis

## MATCHING

- [x] Match ranking
- [x] Multiple matches
- [x] Singleton detection
- [x] Duplicate prevention

## FINAL

- [ ] Final model trained
- [ ] Test prediction generated
- [x] `candidate_pairs.tsv`
- [x] `matching_results.tsv`
- [x] Output validated
- [x] Pipeline reproducible
- [ ] Documentation complete

---

# 10 Major Milestones

Use these as the high-level progress tracker:

- [x] **M1 — Data + EDA**
- [x] **M2 — Preprocessing**
- [x] **M3 — Baseline reproduced**
- [x] **M4 — Blocking improved**
- [ ] **M5 — Similarity features completed**
- [ ] **M6 — ML training pipeline completed**
- [x] **M7 — Threshold + F₀.₅ optimization** (Initial address threshold calibration)
- [x] **M8 — Error analysis + improvements** (Blocking error categorization)
- [ ] **M9 — Final test pipeline + outputs**
- [ ] **M10 — Documentation + submission**
