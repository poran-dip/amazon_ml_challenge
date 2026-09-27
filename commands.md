```bash
# ---- 1. train ----
python -m src.main      --split train --normalize-only --translit --progress
python -m src.blocking  --split train --budget 100 --token-cap 500 --progress
python -m src.features  --split train --progress

python -m src.block_eval --split train --budget 100 --token-cap 500 --progress --no-candidate-recall --no-miss-report --no-key-diagnostics --budget-curve "100"
# -> MACRO RECALL = R

# ---- 2. coverage sweep, ~12 min each, on the NEW feature table ----
python -m src.model --split train --folds 5 --max-folds 1 --coverage R --progress
python -m src.model --split train --folds 5 --max-folds 1 --coverage (0.6*R) --progress
# keep whichever HOLDOUT MACRO F_0.5 is higher -> that value is C

# ---- 3. the real train model ----
python -m src.model     --split train --folds 5 --rounds 1500 --coverage C --progress

# ---- 4. test (writes BOTH tsvs now) ----
python -m src.main      --split test --normalize-only --translit --progress
python -m src.blocking  --split test --budget 100 --token-cap 500 --progress
python -m src.features  --split test --progress
python -m src.model     --split test --folds 5 --rounds 1500 --coverage C --progress
```

PowerShell won't evaluate `(0.6*R)` inline — compute it and paste the number.

**Only if step 3 dies on memory** (not part of the sequence):

```bash
$env:POLARS_MAX_THREADS=6
python -m src.model --split train --folds 5 --rounds 1500 --coverage C --train-rows 4000000 --progress
```

**Validator**

```bash
cd ..\..\student_resource
python3 utils/validate_submission.py --matching ..\output\matching_results.tsv --candidate ..\output\candidate_pairs.tsv --test-dir dataset/test
```
