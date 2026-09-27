# Final Run

Train set output was cleared — test set outputs are as follows.

## Test Set Normalization

```bash
(venv) PS D:\Coding\Projects\working-on\amazon-ml\code\business_entity_resolution> python -m src.main      --split test --normalize-only --translit --progress

[telemetry] telemetry on  run=20260927T213733-16228  ram 15.7GB  avail 9.47GB  work+output 30.69GB  -> D:\Coding\Projects\working-on\amazon-ml\work\timings.jsonl
entity resolution [test]  data=D:\Coding\Projects\working-on\amazon-ml\student_resource
  normalization: text_norm + romanization
  blocking: multi  budget=20
  fuzzy: 16/16 LSH bands (rows=3, detects trigram J=0.5 at 88%)
  scorer: jaccard  threshold=0.15
  normalizing:
    > normalize   chunk=250,000  translit=True  strip_noise=True  forced=False
      > source1   rows_expected=1,732,544
        source1 normalize: 1,732,544/1,732,544 rows  100.0%  32.3k/s  el 53.7s  eta 0.0s  rss 1.26GB  peak 1.29GB  avail 9.18GB  wrote 0.41GB
      < source1   53.9s  1,732,544 rows (32.1k/s)  rows_expected=1,732,544  parquet_mb=424.200  rss 1.26GB  peak 1.29GB  avail 9.18GB  wrote 0.41GB  artefacts+0.04GB
  source1: normalized 1,732,544 rows -> test_source1.parquet
      > source2   rows_expected=4,887,273
        source2 normalize: 4,887,273/4,887,273 rows  100.0%  15.1k/s  el 5m24s  eta 0.0s  rss 2.82GB  peak 3.84GB  avail 7.81GB  wrote 1.69GB
      < source2   5m25s  4,887,273 rows (15.0k/s)  rows_expected=4,887,273  parquet_mb=1,305.500  rss 2.82GB  peak 3.84GB  avail 7.81GB  wrote 1.27GB  artefacts +0.11GB
  source2: normalized 4,887,273 rows -> test_source2.parquet
      > source3   rows_expected=5,082,316
        source3 normalize: 5,082,316/5,082,316 rows  100.0%  16.0k/s  el 5m16s  eta 0.0s  rss 2.82GB  peak 3.84GB  avail 7.87GB  wrote 3.00GB
      < source3   5m17s  5,082,316 rows (16.0k/s)  rows_expected=5,082,316  parquet_mb=1,342.300  rss 2.82GB  peak 2.82GB  avail 7.87GB  wrote 1.31GB  artefacts +0.11GB
  source3: normalized 5,082,316 rows -> test_source3.parquet
    < normalize   11m38s  chunk=250,000  translit=True  strip_noise=True  forced=False  rss 2.82GB  peak 3.84GB  avail 7.87GB  wrote 3.00GB  artefacts +0.26GB
  done in 698.0s (--normalize-only; run `python -m src.blocking --split test` next)
[telemetry] run 20260927T213733-16228 finished in 11m38s   peak rss 3.84GB   disk written 3.00GB   work+output now 30.95GB
```

## Test Set Blocking

```bash
>> python -m src.blocking  --split test --budget 60 --token-cap 500 --merge-buckets 32 --progress

[telemetry] telemetry on  run=20260927T214912-11452  ram 15.7GB  avail 10.60GB  work+output 29.20GB  -> D:\Coding\Projects\working-on\amazon-ml\work\timings.jsonl
blocking [test]  work=D:\Coding\Projects\working-on\amazon-ml\work  budget=60
  exact keys: name_sorted, name_nospace, addr_sorted, name_acronym, name_tokens, addr_tokens, addr_nums, name_prefix, name_charsort, addr_place
  fuzzy: 16/16 LSH bands (rows=3, weight 0.1, detects J=0.5 at 88%)
  bucket caps: name_sorted=2000, name_nospace=2000, addr_sorted=2000, name_acronym=50, name_tokens=500, addr_tokens=500, addr_nums=500, name_prefix=2000, name_charsort=2000, addr_place=2000, 16 bands=100
    > blocking   split=test  budget=60  exact_keys=10  bands=16
  removed 384 stale partial(s) from a previous run
      > read-source1   keys=26
      < read-source1   0.3s  1,732,544 rows (5.01M/s)  keys=26  resident_mb=602.700  rss 1.19GB  peak 1.19GB  avail 9.44GB  wrote 0.00GB  artefacts +0.00GB
  source1: 1,732,544 rows resident (603 MB)
      > emit   keys=26
        > emit:name_sorted   cap=2,000
          > targets:name_sorted   cap=2,000
          < targets:name_sorted   1.5s  5,354,905 rows (3.66M/s)  cap=2,000  rss 2.48GB  peak 2.67GB  avail 8.14GB  wrote 0.00GB  artefacts +0.00GB
          > targets-join:name_sorted
          < targets-join:name_sorted   1.0s  9,969,578 rows (10.04M/s)  resident_mb=382.100  buckets_kept=5,354,905  buckets_dropped=0  rss 3.19GB  peak 3.19GB  avail 7.42GB  wrote 0.00GB  artefacts +0.00GB
    name_sorted    targets    9,969,578 rows   0/5,354,905 buckets over cap 2,000 dropped
          emit name_sorted: 65,539,080/65,539,080 pairs  100.0%  2.56M/s  el 25.6s  eta 0.0s  rss 3.00GB  peak 3.00GB  avail 7.52GB  wrote 0.35GB              
        - name_sorted: emitted   pairs=65,539,080  bucket_files=32  parquet_mb=357.400  running_total=65,539,080
    name_sorted      65,539,080 raw pairs -> 32 bucket files   (running total 65,539,080)
        < emit:name_sorted   28.6s  cap=2,000  files=32  running_pairs=65,539,080  rss 3.00GB  peak 3.00GB  avail 7.52GB  wrote 0.35GB  artefacts +0.35GB
        keys: 1/26 keys   3.8%  0/s  el 28.6s  eta 11m54s  pairs 65,539,080  rss 3.00GB  peak 3.00GB  avail 7.52GB  wrote 0.35GB
        > emit:name_nospace   cap=2,000
          > targets:name_nospace   cap=2,000
          < targets:name_nospace   1.7s  5,445,399 rows (3.21M/s)  cap=2,000  rss 2.62GB  peak 3.14GB  avail 7.89GB  wrote 0.00GB  artefacts +0.00GB
          > targets-join:name_nospace
          < targets-join:name_nospace   1.0s  9,969,578 rows (10.13M/s)  resident_mb=367.400  buckets_kept=5,445,399  buckets_dropped=0  rss 3.30GB  peak 3.49GB  avail 7.19GB  wrote 0.00GB  artefacts +0.00GB
    name_nospace   targets    9,969,578 rows   0/5,445,399 buckets over cap 2,000 dropped
          emit name_nospace: 64,905,392/64,905,392 pairs  100.0%  2.58M/s  el 25.1s  eta 0.0s  rss 2.93GB  peak 3.49GB  avail 7.56GB  wrote 0.69GB 
        - name_nospace: emitted   pairs=64,905,392  bucket_files=32  parquet_mb=354.300  running_total=130,444,472
    name_nospace     64,905,392 raw pairs -> 32 bucket files   (running total 130,444,472)
        < emit:name_nospace   28.3s  cap=2,000  files=32  running_pairs=130,444,472  rss 2.93GB  peak 3.49GB  avail 7.56GB  wrote 0.35GB  artefacts +0.35GB
        keys: 2/26 keys   7.7%  0/s  el 56.8s  eta 11m22s  pairs 130,444,472  rss 2.93GB  peak 3.49GB  avail 7.56GB  wrote 0.69GB
        > emit:addr_sorted   cap=2,000
          > targets:addr_sorted   cap=2,000
          < targets:addr_sorted   1.8s  5,210,462 rows (2.86M/s)  cap=2,000  rss 2.88GB  peak 3.60GB  avail 7.64GB  wrote 0.00GB  artefacts +0.00GB
          > targets-join:addr_sorted
          < targets-join:addr_sorted   1.2s  9,669,670 rows (7.92M/s)  resident_mb=573.000  buckets_kept=5,210,452  buckets_dropped=10  rss 3.71GB  peak 4.08GB  avail 6.82GB  wrote 0.00GB  artefacts +0.00GB
    addr_sorted    targets    9,669,670 rows   10/5,210,462 buckets over cap 2,000 dropped
          emit addr_sorted: 16,735,249/16,735,249 pairs  100.0%  1.95M/s  el 8.6s  eta 0.0s  rss 3.28GB  peak 4.08GB  avail 7.25GB  wrote 0.79GB              
        - addr_sorted: emitted   pairs=16,735,249  bucket_files=32  parquet_mb=92.800  running_total=147,179,721
    addr_sorted      16,735,249 raw pairs -> 32 bucket files   (running total 147,179,721)
        < emit:addr_sorted   12.0s  cap=2,000  files=32  running_pairs=147,179,721  rss 3.28GB  peak 4.08GB  avail 7.25GB  wrote 0.09GB  artefacts +0.09GB
        keys: 3/26 keys  11.5%  0/s  el 1m08s  eta 8m48s  pairs 147,179,721  rss 3.28GB  peak 4.08GB  avail 7.25GB  wrote 0.79GB
        > emit:name_acronym   cap=50
          > targets:name_acronym   cap=50
          < targets:name_acronym   0.7s  554,656 rows (795.4k/s)  cap=50  rss 2.40GB  peak 3.28GB  avail 8.13GB  wrote 0.00GB  artefacts +0.00GB
          > targets-join:name_acronym
          < targets-join:name_acronym   0.5s  1,787,093 rows (3.52M/s)  resident_mb=41.500  buckets_kept=537,333  buckets_dropped=17,323  rss 2.78GB  peak 2.78GB  avail 7.76GB  wrote 0.00GB  artefacts +0.00GB
    name_acronym   targets    1,787,093 rows   17,323/554,656 buckets over cap 50 dropped
          emit name_acronym: 2,599,195/2,599,195 pairs  100.0%  1.53M/s  el 1.7s  eta 0.0s  rss 1.95GB  peak 4.08GB  avail 8.59GB  wrote 0.81GB             
        - name_acronym: emitted   pairs=2,599,195  bucket_files=32  parquet_mb=22.400  running_total=149,778,916
    name_acronym      2,599,195 raw pairs -> 32 bucket files   (running total 149,778,916)
        < emit:name_acronym   3.0s  cap=50  files=32  running_pairs=149,778,916  rss 1.95GB  peak 3.28GB  avail 8.59GB  wrote 0.02GB  artefacts +0.02GB
        keys: 4/26 keys  15.4%  0/s  el 1m11s  eta 6m35s  pairs 149,778,916  rss 1.95GB  peak 4.08GB  avail 8.59GB  wrote 0.81GB
        > emit:band_00   cap=100
          > targets:band_00   cap=100
          < targets:band_00   1.1s  2,107,277 rows (1.93M/s)  cap=100  rss 2.98GB  peak 2.98GB  avail 7.56GB  wrote 0.00GB  artefacts +0.00GB
          > targets-join:band_00
          < targets-join:band_00   0.9s  6,575,373 rows (7.35M/s)  resident_mb=248.300  buckets_kept=2,097,121  buckets_dropped=10,156  rss 3.35GB  peak 3.35GB  avail 7.21GB  wrote 0.00GB  artefacts +0.00GB
    band_00        targets    6,575,373 rows   10,156/2,107,277 buckets over cap 100 dropped
          emit band_00: 23,225,424/23,225,424 pairs  100.0%  2.27M/s  el 10.2s  eta 0.0s  rss 2.68GB  peak 4.08GB  avail 7.81GB  wrote 0.97GB              
        - band_00: emitted   pairs=23,225,424  bucket_files=32  parquet_mb=165.000  running_total=173,004,340
    band_00          23,225,424 raw pairs -> 32 bucket files   (running total 173,004,340)
        < emit:band_00   12.4s  cap=100  files=32  running_pairs=173,004,340  rss 2.68GB  peak 2.68GB  avail 7.81GB  wrote 0.16GB  artefacts +0.16GB
        keys: 5/26 keys  19.2%  0/s  el 1m24s  eta 5m54s  pairs 173,004,340  rss 2.68GB  peak 4.08GB  avail 7.81GB  wrote 0.97GB
        > emit:band_01   cap=100
          > targets:band_01   cap=100
          < targets:band_01   1.2s  2,330,025 rows (2.01M/s)  cap=100  rss 3.20GB  peak 3.20GB  avail 7.30GB  wrote 0.00GB  artefacts +0.00GB
          > targets-join:band_01
          < targets-join:band_01   0.9s  6,928,619 rows (7.89M/s)  resident_mb=261.700  buckets_kept=2,320,227  buckets_dropped=9,798  rss 3.37GB  peak 3.37GB avail 7.13GB  wrote 0.00GB  artefacts +0.00GB
    band_01        targets    6,928,619 rows   9,798/2,330,025 buckets over cap 100 dropped
          emit band_01: 22,396,703/22,396,703 pairs  100.0%  2.22M/s  el 10.1s  eta 0.0s  rss 2.73GB  peak 4.08GB  avail 7.73GB  wrote 1.13GB              
        - band_01: emitted   pairs=22,396,703  bucket_files=32  parquet_mb=160.600  running_total=195,401,043
    band_01          22,396,703 raw pairs -> 32 bucket files   (running total 195,401,043)
        < emit:band_01   12.3s  cap=100  files=32  running_pairs=195,401,043  rss 2.73GB  peak 2.73GB  avail 7.73GB  wrote 0.16GB  artefacts +0.16GB
        keys: 6/26 keys  23.1%  0/s  el 1m36s  eta 5m22s  pairs 195,401,043  rss 2.73GB  peak 4.08GB  avail 7.73GB  wrote 1.13GB
        > emit:band_02   cap=100
          > targets:band_02   cap=100
          < targets:band_02   1.2s  2,254,898 rows (1.91M/s)  cap=100  rss 3.27GB  peak 3.27GB  avail 7.21GB  wrote 0.00GB  artefacts +0.00GB
          > targets-join:band_02
          < targets-join:band_02   0.9s  6,850,975 rows (7.61M/s)  resident_mb=258.800  buckets_kept=2,245,282  buckets_dropped=9,616  rss 3.30GB  peak 3.30GB avail 7.19GB  wrote 0.00GB  artefacts +0.00GB
    band_02        targets    6,850,975 rows   9,616/2,254,898 buckets over cap 100 dropped
          emit band_02: 23,559,121/23,559,121 pairs  100.0%  2.25M/s  el 10.5s  eta 0.0s  rss 2.62GB  peak 4.08GB  avail 7.83GB  wrote 1.29GB              
        - band_02: emitted   pairs=23,559,121  bucket_files=32  parquet_mb=167.600  running_total=218,960,164
    band_02          23,559,121 raw pairs -> 32 bucket files   (running total 218,960,164)
        < emit:band_02   12.8s  cap=100  files=32  running_pairs=218,960,164  rss 2.62GB  peak 2.73GB  avail 7.83GB  wrote 0.16GB  artefacts +0.16GB
        keys: 7/26 keys  26.9%  0/s  el 1m49s  eta 4m56s  pairs 218,960,164  rss 2.62GB  peak 4.08GB  avail 7.83GB  wrote 1.29GB
        > emit:band_03   cap=100
          > targets:band_03   cap=100
          < targets:band_03   1.1s  2,238,163 rows (2.05M/s)  cap=100  rss 2.86GB  peak 2.86GB  avail 7.60GB  wrote 0.00GB  artefacts +0.00GB
          > targets-join:band_03
          < targets-join:band_03   0.9s  6,835,169 rows (7.25M/s)  resident_mb=258.100  buckets_kept=2,228,488  buckets_dropped=9,675  rss 3.06GB  peak 3.06GB avail 7.40GB  wrote 0.00GB  artefacts +0.00GB
    band_03        targets    6,835,169 rows   9,675/2,238,163 buckets over cap 100 dropped
          emit band_03: 23,034,335/23,034,335 pairs  100.0%  2.26M/s  el 10.2s  eta 0.0s  rss 2.79GB  peak 4.08GB  avail 7.66GB  wrote 1.45GB              
        - band_03: emitted   pairs=23,034,335  bucket_files=32  parquet_mb=164.300  running_total=241,994,499
    band_03          23,034,335 raw pairs -> 32 bucket files   (running total 241,994,499)
        < emit:band_03   12.4s  cap=100  files=32  running_pairs=241,994,499  rss 2.79GB  peak 2.79GB  avail 7.66GB  wrote 0.16GB  artefacts +0.16GB
        keys: 8/26 keys  30.8%  0/s  el 2m01s  eta 4m34s  pairs 241,994,499  rss 2.79GB  peak 4.08GB  avail 7.66GB  wrote 1.45GB
        > emit:band_04   cap=100
          > targets:band_04   cap=100
          < targets:band_04   1.2s  2,304,707 rows (1.96M/s)  cap=100  rss 3.16GB  peak 3.16GB  avail 7.28GB  wrote 0.00GB  artefacts +0.00GB
          > targets-join:band_04
          < targets-join:band_04   1.2s  6,988,494 rows (5.97M/s)  resident_mb=264.300  buckets_kept=2,294,965  buckets_dropped=9,742  rss 3.32GB  peak 3.32GB avail 7.13GB  wrote 0.00GB  artefacts +0.00GB
    band_04        targets    6,988,494 rows   9,742/2,304,707 buckets over cap 100 dropped
          emit band_04: 22,926,953/22,926,953 pairs  100.0%  2.22M/s  el 10.3s  eta 0.0s  rss 2.73GB  peak 4.08GB  avail 7.69GB  wrote 1.61GB              
        - band_04: emitted   pairs=22,926,953  bucket_files=32  parquet_mb=164.200  running_total=264,921,452
    band_04          22,926,953 raw pairs -> 32 bucket files   (running total 264,921,452)
        < emit:band_04   12.9s  cap=100  files=32  running_pairs=264,921,452  rss 2.73GB  peak 2.79GB  avail 7.69GB  wrote 0.16GB  artefacts +0.16GB
        keys: 9/26 keys  34.6%  0/s  el 2m14s  eta 4m14s  pairs 264,921,452  rss 2.73GB  peak 4.08GB  avail 7.69GB  wrote 1.61GB
        > emit:band_05   cap=100
          > targets:band_05   cap=100
          < targets:band_05   1.1s  2,186,391 rows (1.92M/s)  cap=100  rss 3.09GB  peak 3.09GB  avail 7.35GB  wrote 0.00GB  artefacts +0.00GB
          > targets-join:band_05
          < targets-join:band_05   0.9s  6,631,689 rows (7.59M/s)  resident_mb=250.400  buckets_kept=2,176,638  buckets_dropped=9,753  rss 3.31GB  peak 3.31GB avail 7.14GB  wrote 0.00GB  artefacts +0.00GB
    band_05        targets    6,631,689 rows   9,753/2,186,391 buckets over cap 100 dropped
          emit band_05: 22,040,262/22,040,262 pairs  100.0%  2.24M/s  el 9.9s  eta 0.0s  rss 2.72GB  peak 4.08GB  avail 7.71GB  wrote 1.76GB              
        - band_05: emitted   pairs=22,040,262  bucket_files=32  parquet_mb=158.100  running_total=286,961,714
    band_05          22,040,262 raw pairs -> 32 bucket files   (running total 286,961,714)
        < emit:band_05   12.1s  cap=100  files=32  running_pairs=286,961,714  rss 2.72GB  peak 2.73GB  avail 7.71GB  wrote 0.15GB  artefacts +0.15GB
        keys: 10/26 keys  38.5%  0/s  el 2m26s  eta 3m54s  pairs 286,961,714  rss 2.72GB  peak 4.08GB  avail 7.71GB  wrote 1.76GB
        > emit:band_06   cap=100
          > targets:band_06   cap=100
          < targets:band_06   1.2s  2,317,948 rows (1.99M/s)  cap=100  rss 3.02GB  peak 3.02GB  avail 7.44GB  wrote 0.00GB  artefacts +0.00GB
          > targets-join:band_06
          < targets-join:band_06   0.9s  6,947,878 rows (7.58M/s)  resident_mb=262.700  buckets_kept=2,308,177  buckets_dropped=9,771  rss 3.13GB  peak 3.13GB avail 7.31GB  wrote 0.00GB  artefacts +0.00GB
    band_06        targets    6,947,878 rows   9,771/2,317,948 buckets over cap 100 dropped
          emit band_06: 23,308,858/23,308,858 pairs  100.0%  2.23M/s  el 10.5s  eta 0.0s  rss 2.76GB  peak 4.08GB  avail 7.69GB  wrote 1.93GB              
        - band_06: emitted   pairs=23,308,858  bucket_files=32  parquet_mb=166.400  running_total=310,270,572
    band_06          23,308,858 raw pairs -> 32 bucket files   (running total 310,270,572)
        < emit:band_06   12.7s  cap=100  files=32  running_pairs=310,270,572  rss 2.76GB  peak 2.76GB  avail 7.69GB  wrote 0.16GB  artefacts +0.16GB
        keys: 11/26 keys  42.3%  0/s  el 2m39s  eta 3m37s  pairs 310,270,572  rss 2.76GB  peak 4.08GB  avail 7.69GB  wrote 1.93GB
        > emit:band_07   cap=100
          > targets:band_07   cap=100
          < targets:band_07   1.1s  2,170,820 rows (1.93M/s)  cap=100  rss 3.08GB  peak 3.08GB  avail 7.36GB  wrote 0.00GB  artefacts +0.00GB
          > targets-join:band_07
          < targets-join:band_07   0.9s  6,723,740 rows (7.73M/s)  resident_mb=254.300  buckets_kept=2,160,684  buckets_dropped=10,136  rss 3.46GB  peak 3.46GB  avail 7.02GB  wrote 0.00GB  artefacts +0.00GB
    band_07        targets    6,723,740 rows   10,136/2,170,820 buckets over cap 100 dropped
          emit band_07: 23,547,142/23,547,142 pairs  100.0%  2.26M/s  el 10.4s  eta 0.0s  rss 2.72GB  peak 4.08GB  avail 7.73GB  wrote 2.09GB              
        - band_07: emitted   pairs=23,547,142  bucket_files=32  parquet_mb=166.900  running_total=333,817,714
    band_07          23,547,142 raw pairs -> 32 bucket files   (running total 333,817,714)
        < emit:band_07   12.7s  cap=100  files=32  running_pairs=333,817,714  rss 2.72GB  peak 2.76GB  avail 7.73GB  wrote 0.16GB  artefacts +0.16GB
        keys: 12/26 keys  46.2%  0/s  el 2m52s  eta 3m21s  pairs 333,817,714  rss 2.72GB  peak 4.08GB  avail 7.73GB  wrote 2.09GB
        > emit:band_08   cap=100
          > targets:band_08   cap=100
          < targets:band_08   1.2s  2,165,574 rows (1.83M/s)  cap=100  rss 2.91GB  peak 2.91GB  avail 7.55GB  wrote 0.00GB  artefacts +0.00GB
          > targets-join:band_08
          < targets-join:band_08   0.9s  6,667,305 rows (7.20M/s)  resident_mb=251.800  buckets_kept=2,155,627  buckets_dropped=9,947  rss 3.36GB  peak 3.36GB avail 7.10GB  wrote 0.00GB  artefacts +0.00GB
    band_08        targets    6,667,305 rows   9,947/2,165,574 buckets over cap 100 dropped
          emit band_08: 23,226,418/23,226,418 pairs  100.0%  2.24M/s  el 10.4s  eta 0.0s  rss 2.64GB  peak 4.08GB  avail 7.80GB  wrote 2.25GB              
        - band_08: emitted   pairs=23,226,418  bucket_files=32  parquet_mb=165.700  running_total=357,044,132
    band_08          23,226,418 raw pairs -> 32 bucket files   (running total 357,044,132)
        < emit:band_08   12.8s  cap=100  files=32  running_pairs=357,044,132  rss 2.64GB  peak 2.72GB  avail 7.80GB  wrote 0.16GB  artefacts +0.16GB
        keys: 13/26 keys  50.0%  0/s  el 3m05s  eta 3m05s  pairs 357,044,132  rss 2.64GB  peak 4.08GB  avail 7.80GB  wrote 2.25GB
        > emit:band_09   cap=100
          > targets:band_09   cap=100
          < targets:band_09   1.1s  2,360,179 rows (2.09M/s)  cap=100  rss 2.96GB  peak 2.96GB  avail 7.49GB  wrote 0.00GB  artefacts +0.00GB
          > targets-join:band_09
          < targets-join:band_09   1.2s  7,067,495 rows (6.01M/s)  resident_mb=267.300  buckets_kept=2,350,325  buckets_dropped=9,854  rss 3.23GB  peak 3.23GB avail 7.22GB  wrote 0.00GB  artefacts +0.00GB
    band_09        targets    7,067,495 rows   9,854/2,360,179 buckets over cap 100 dropped
          emit band_09: 23,794,721/23,794,721 pairs  100.0%  2.24M/s  el 10.6s  eta 0.0s  rss 2.76GB  peak 4.08GB  avail 7.66GB  wrote 2.42GB              
        - band_09: emitted   pairs=23,794,721  bucket_files=32  parquet_mb=169.200  running_total=380,838,853
    band_09          23,794,721 raw pairs -> 32 bucket files   (running total 380,838,853)
        < emit:band_09   13.2s  cap=100  files=32  running_pairs=380,838,853  rss 2.76GB  peak 2.76GB  avail 7.66GB  wrote 0.17GB  artefacts +0.17GB
        keys: 14/26 keys  53.8%  0/s  el 3m18s  eta 2m49s  pairs 380,838,853  rss 2.76GB  peak 4.08GB  avail 7.66GB  wrote 2.42GB
        > emit:band_10   cap=100
          > targets:band_10   cap=100
          < targets:band_10   1.3s  2,373,840 rows (1.88M/s)  cap=100  rss 3.21GB  peak 3.21GB  avail 7.24GB  wrote 0.00GB  artefacts +0.00GB
          > targets-join:band_10
          < targets-join:band_10   1.1s  7,110,697 rows (6.26M/s)  resident_mb=268.500  buckets_kept=2,364,121  buckets_dropped=9,719  rss 3.26GB  peak 3.26GB avail 7.20GB  wrote 0.00GB  artefacts +0.00GB
    band_10        targets    7,110,697 rows   9,719/2,373,840 buckets over cap 100 dropped
          emit band_10: 22,878,096/22,878,096 pairs  100.0%  2.19M/s  el 10.4s  eta 0.0s  rss 2.73GB  peak 4.08GB  avail 7.69GB  wrote 2.58GB              
        - band_10: emitted   pairs=22,878,096  bucket_files=32  parquet_mb=164.400  running_total=403,716,949
    band_10          22,878,096 raw pairs -> 32 bucket files   (running total 403,716,949)
        < emit:band_10   13.1s  cap=100  files=32  running_pairs=403,716,949  rss 2.73GB  peak 2.76GB  avail 7.69GB  wrote 0.16GB  artefacts +0.16GB
        keys: 15/26 keys  57.7%  0/s  el 3m31s  eta 2m34s  pairs 403,716,949  rss 2.73GB  peak 4.08GB  avail 7.69GB  wrote 2.58GB
        > emit:band_11   cap=100
          > targets:band_11   cap=100
          < targets:band_11   1.2s  2,229,407 rows (1.91M/s)  cap=100  rss 3.18GB  peak 3.18GB  avail 7.26GB  wrote 0.00GB  artefacts +0.00GB
          > targets-join:band_11
          < targets-join:band_11   1.1s  6,837,693 rows (6.05M/s)  resident_mb=258.700  buckets_kept=2,219,631  buckets_dropped=9,776  rss 3.29GB  peak 3.29GB avail 7.16GB  wrote 0.00GB  artefacts +0.00GB
    band_11        targets    6,837,693 rows   9,776/2,229,407 buckets over cap 100 dropped
          emit band_11: 23,198,620/23,198,620 pairs  100.0%  2.22M/s  el 10.4s  eta 0.0s  rss 2.74GB  peak 4.08GB  avail 7.69GB  wrote 2.74GB              
        - band_11: emitted   pairs=23,198,620  bucket_files=32  parquet_mb=165.400  running_total=426,915,569
    band_11          23,198,620 raw pairs -> 32 bucket files   (running total 426,915,569)
        < emit:band_11   13.0s  cap=100  files=32  running_pairs=426,915,569  rss 2.74GB  peak 2.74GB  avail 7.69GB  wrote 0.16GB  artefacts +0.16GB
        keys: 16/26 keys  61.5%  0/s  el 3m44s  eta 2m20s  pairs 426,915,569  rss 2.74GB  peak 4.08GB  avail 7.69GB  wrote 2.74GB
        > emit:band_12   cap=100
          > targets:band_12   cap=100
          < targets:band_12   1.1s  2,192,497 rows (1.94M/s)  cap=100  rss 3.18GB  peak 3.18GB  avail 7.27GB  wrote 0.00GB  artefacts +0.00GB
          > targets-join:band_12
          < targets-join:band_12   0.9s  6,681,306 rows (7.68M/s)  resident_mb=252.100  buckets_kept=2,182,559  buckets_dropped=9,938  rss 3.41GB  peak 3.41GB avail 7.05GB  wrote 0.00GB  artefacts +0.00GB
    band_12        targets    6,681,306 rows   9,938/2,192,497 buckets over cap 100 dropped
          emit band_12: 22,793,630/22,793,630 pairs  100.0%  2.25M/s  el 10.1s  eta 0.0s  rss 2.72GB  peak 4.08GB  avail 7.72GB  wrote 2.90GB              
        - band_12: emitted   pairs=22,793,630  bucket_files=32  parquet_mb=162.100  running_total=449,709,199
    band_12          22,793,630 raw pairs -> 32 bucket files   (running total 449,709,199)
        < emit:band_12   12.3s  cap=100  files=32  running_pairs=449,709,199  rss 2.72GB  peak 2.74GB  avail 7.72GB  wrote 0.16GB  artefacts +0.16GB
        keys: 17/26 keys  65.4%  0/s  el 3m56s  eta 2m05s  pairs 449,709,199  rss 2.72GB  peak 4.08GB  avail 7.72GB  wrote 2.90GB
        > emit:band_13   cap=100
          > targets:band_13   cap=100
          < targets:band_13   1.1s  2,135,697 rows (1.90M/s)  cap=100  rss 3.22GB  peak 3.22GB  avail 7.23GB  wrote 0.00GB  artefacts +0.00GB
          > targets-join:band_13
          < targets-join:band_13   0.9s  6,603,260 rows (7.16M/s)  resident_mb=249.500  buckets_kept=2,125,634  buckets_dropped=10,063  rss 3.43GB  peak 3.43GB  avail 7.05GB  wrote 0.00GB  artefacts +0.00GB
    band_13        targets    6,603,260 rows   10,063/2,135,697 buckets over cap 100 dropped
          emit band_13: 22,330,846/22,330,846 pairs  100.0%  2.25M/s  el 9.9s  eta 0.0s  rss 2.83GB  peak 4.08GB  avail 7.62GB  wrote 3.05GB              
        - band_13: emitted   pairs=22,330,846  bucket_files=32  parquet_mb=159.700  running_total=472,040,045
    band_13          22,330,846 raw pairs -> 32 bucket files   (running total 472,040,045)
        < emit:band_13   12.2s  cap=100  files=32  running_pairs=472,040,045  rss 2.83GB  peak 2.83GB  avail 7.62GB  wrote 0.16GB  artefacts +0.16GB
        keys: 18/26 keys  69.2%  0/s  el 4m08s  eta 1m50s  pairs 472,040,045  rss 2.83GB  peak 4.08GB  avail 7.62GB  wrote 3.05GB
        > emit:band_14   cap=100
          > targets:band_14   cap=100
          < targets:band_14   1.1s  2,108,182 rows (1.94M/s)  cap=100  rss 3.19GB  peak 3.19GB  avail 7.27GB  wrote 0.00GB  artefacts +0.00GB
          > targets-join:band_14
          < targets-join:band_14   0.9s  6,553,680 rows (7.16M/s)  resident_mb=247.500  buckets_kept=2,098,126  buckets_dropped=10,056  rss 3.38GB  peak 3.38GB  avail 7.04GB  wrote 0.00GB  artefacts +0.00GB
    band_14        targets    6,553,680 rows   10,056/2,108,182 buckets over cap 100 dropped
          emit band_14: 23,550,612/23,550,612 pairs  100.0%  2.27M/s  el 10.4s  eta 0.0s  rss 2.71GB  peak 4.08GB  avail 7.73GB  wrote 3.22GB              
        - band_14: emitted   pairs=23,550,612  bucket_files=32  parquet_mb=166.700  running_total=495,590,657
    band_14          23,550,612 raw pairs -> 32 bucket files   (running total 495,590,657)
        < emit:band_14   12.6s  cap=100  files=32  running_pairs=495,590,657  rss 2.71GB  peak 2.83GB  avail 7.73GB  wrote 0.16GB  artefacts +0.16GB
        keys: 19/26 keys  73.1%  0/s  el 4m21s  eta 1m36s  pairs 495,590,657  rss 2.71GB  peak 4.08GB  avail 7.73GB  wrote 3.22GB
        > emit:band_15   cap=100
          > targets:band_15   cap=100
          < targets:band_15   1.2s  2,095,900 rows (1.81M/s)  cap=100  rss 3.14GB  peak 3.14GB  avail 7.32GB  wrote 0.00GB  artefacts +0.00GB
          > targets-join:band_15
          < targets-join:band_15   0.9s  6,517,288 rows (7.52M/s)  resident_mb=246.400  buckets_kept=2,085,732  buckets_dropped=10,168  rss 3.31GB  peak 3.31GB  avail 7.16GB  wrote 0.00GB  artefacts +0.00GB
    band_15        targets    6,517,288 rows   10,168/2,095,900 buckets over cap 100 dropped
          emit band_15: 23,016,375/23,016,375 pairs  100.0%  2.29M/s  el 10.1s  eta 0.0s  rss 2.62GB  peak 4.08GB  avail 7.81GB  wrote 3.38GB              
        - band_15: emitted   pairs=23,016,375  bucket_files=32  parquet_mb=163.700  running_total=518,607,032
    band_15          23,016,375 raw pairs -> 32 bucket files   (running total 518,607,032)
        < emit:band_15   12.4s  cap=100  files=32  running_pairs=518,607,032  rss 2.62GB  peak 2.71GB  avail 7.81GB  wrote 0.16GB  artefacts +0.16GB
        keys: 20/26 keys  76.9%  0/s  el 4m33s  eta 1m22s  pairs 518,607,032  rss 2.62GB  peak 4.08GB  avail 7.81GB  wrote 3.38GB
        > emit:name_tokens   cap=500
          > targets:name_tokens   cap=500
          < targets:name_tokens   2.2s  1,187,436 rows (531.9k/s)  cap=500  rss 3.80GB  peak 3.80GB  avail 6.67GB  wrote 0.00GB  artefacts +0.00GB
          > targets-join:name_tokens
          < targets-join:name_tokens   2.1s  5,811,208 rows (2.78M/s)  resident_mb=156.000  buckets_kept=1,183,455  buckets_dropped=3,981  rss 4.10GB  peak 4.67GB  avail 6.40GB  wrote 0.00GB  artefacts +0.00GB
    name_tokens    targets    5,811,208 rows   3,981/1,187,436 buckets over cap 500 dropped
          emit name_tokens: 105,905,053/105,905,053 pairs  100.0%  2.82M/s  el 37.6s  eta 0.0s  rss 2.95GB  peak 4.67GB  avail 7.48GB  wrote 4.01GB  
        - name_tokens: emitted   pairs=105,905,053  bucket_files=32  parquet_mb=653.300  running_total=624,512,085
    name_tokens     105,905,053 raw pairs -> 32 bucket files   (running total 624,512,085)
        < emit:name_tokens   42.5s  cap=500  files=32  running_pairs=624,512,085  rss 2.95GB  peak 4.67GB  avail 7.48GB  wrote 0.64GB  artefacts +0.64GB
        keys: 21/26 keys  80.8%  0/s  el 5m16s  eta 1m15s  pairs 624,512,085  rss 2.95GB  peak 4.67GB  avail 7.48GB  wrote 4.01GB
        > emit:addr_tokens   cap=500
          > targets:addr_tokens   cap=500
          < targets:addr_tokens   4.9s  678,188 rows (139.4k/s)  cap=500  rss 4.84GB  peak 5.75GB  avail 5.66GB  wrote 0.00GB  artefacts +0.00GB
          > targets-join:addr_tokens
          < targets-join:addr_tokens   3.9s  8,749,401 rows (2.24M/s)  resident_mb=238.100  buckets_kept=670,829  buckets_dropped=7,359  rss 6.12GB  peak 6.12GB  avail 4.42GB  wrote 0.00GB  artefacts +0.00GB
    addr_tokens    targets    8,749,401 rows   7,359/678,188 buckets over cap 500 dropped
          emit addr_tokens: 231,145,906/231,145,906 pairs  100.0%  2.72M/s  el 1m24s  eta 0.0s  rss 3.54GB  peak 6.41GB  avail 6.95GB  wrote 5.40GB  
        - addr_tokens: emitted   pairs=231,145,906  bucket_files=32  parquet_mb=1,422.500  running_total=855,657,991
    addr_tokens     231,145,906 raw pairs -> 32 bucket files   (running total 855,657,991)
        < emit:addr_tokens   1m35s  cap=500  files=32  running_pairs=855,657,991  rss 3.54GB  peak 6.41GB  avail 6.95GB  wrote 1.39GB  artefacts +1.39GB
        keys: 22/26 keys  84.6%  0/s  el 6m51s  eta 1m14s  pairs 855,657,991  rss 3.54GB  peak 6.41GB  avail 6.95GB  wrote 5.40GB
        > emit:addr_nums   cap=500
          > targets:addr_nums   cap=500
          < targets:addr_nums   0.8s  91,310 rows (118.4k/s)  cap=500  rss 2.23GB  peak 3.54GB  avail 8.29GB  wrote 0.00GB  artefacts +0.00GB
          > targets-join:addr_nums
          < targets-join:addr_nums   0.6s  2,383,868 rows (4.30M/s)  resident_mb=54.000  buckets_kept=88,908  buckets_dropped=2,402  rss 3.26GB  peak 3.26GB  avail 7.25GB  wrote 0.00GB  artefacts +0.00GB
    addr_nums      targets    2,383,868 rows   2,402/91,310 buckets over cap 500 dropped
          emit addr_nums: 69,683,117/69,683,117 pairs  100.0%  2.90M/s  el 24.0s  eta 0.0s  rss 2.44GB  peak 6.41GB  avail 8.07GB  wrote 5.80GB              
        - addr_nums: emitted   pairs=69,683,117  bucket_files=32  parquet_mb=410.700  running_total=925,341,108
    addr_nums        69,683,117 raw pairs -> 32 bucket files   (running total 925,341,108)
        < emit:addr_nums   25.5s  cap=500  files=32  running_pairs=925,341,108  rss 2.44GB  peak 3.54GB  avail 8.07GB  wrote 0.40GB  artefacts +0.40GB
        keys: 23/26 keys  88.5%  0/s  el 7m17s  eta 57.0s  pairs 925,341,108  rss 2.44GB  peak 6.41GB  avail 8.07GB  wrote 5.80GB
        > emit:name_prefix   cap=2,000
          > targets:name_prefix   cap=2,000
          < targets:name_prefix   0.9s  2,595,007 rows (2.92M/s)  cap=2,000  rss 2.75GB  peak 2.75GB  avail 7.78GB  wrote 0.00GB  artefacts +0.00GB
          > targets-join:name_prefix
          < targets-join:name_prefix   0.5s  7,852,182 rows (14.33M/s)  resident_mb=245.700  buckets_kept=2,594,965  buckets_dropped=42  rss 2.82GB  peak 2.82GB  avail 7.72GB  wrote 0.00GB  artefacts +0.00GB
    name_prefix    targets    7,852,182 rows   42/2,595,007 buckets over cap 2,000 dropped
          emit name_prefix: 98,102,089/98,102,089 pairs  100.0%  2.71M/s  el 36.2s  eta 0.0s  rss 3.06GB  peak 6.41GB  avail 7.48GB  wrote 6.30GB              
        - name_prefix: emitted   pairs=98,102,089  bucket_files=32  parquet_mb=507.700  running_total=1,023,443,197
    name_prefix      98,102,089 raw pairs -> 32 bucket files   (running total 1,023,443,197)
        < emit:name_prefix   37.8s  cap=2,000  files=32  running_pairs=1,023,443,197  rss 3.06GB  peak 3.06GB  avail 7.48GB  wrote 0.50GB  artefacts +0.50GB
        keys: 24/26 keys  92.3%  0/s  el 7m54s  eta 39.6s  pairs 1,023,443,197  rss 3.06GB  peak 6.41GB  avail 7.48GB  wrote 6.30GB
        > emit:name_charsort   cap=2,000
          > targets:name_charsort   cap=2,000
          < targets:name_charsort   1.8s  5,142,765 rows (2.88M/s)  cap=2,000  rss 2.54GB  peak 3.06GB  avail 7.97GB  wrote 0.00GB  artefacts +0.00GB
          > targets-join:name_charsort
          < targets-join:name_charsort   1.4s  9,969,578 rows (7.20M/s)  resident_mb=367.400  buckets_kept=5,142,765  buckets_dropped=0  rss 3.36GB  peak 3.36GB  avail 7.12GB  wrote 0.00GB  artefacts +0.00GB
    name_charsort  targets    9,969,578 rows   0/5,142,765 buckets over cap 2,000 dropped
          emit name_charsort: 67,908,469/67,908,469 pairs  100.0%  2.51M/s  el 27.0s  eta 0.0s  rss 3.12GB  peak 6.41GB  avail 7.49GB  wrote 6.67GB  
        - name_charsort: emitted   pairs=67,908,469  bucket_files=32  parquet_mb=375.300  running_total=1,091,351,666
    name_charsort    67,908,469 raw pairs -> 32 bucket files   (running total 1,091,351,666)
        < emit:name_charsort   30.5s  cap=2,000  files=32  running_pairs=1,091,351,666  rss 3.12GB  peak 3.12GB  avail 7.49GB  wrote 0.37GB  artefacts +0.37GB
        keys: 25/26 keys  96.2%  0/s  el 8m25s  eta 20.2s  pairs 1,091,351,666  rss 3.12GB  peak 6.41GB  avail 7.49GB  wrote 6.67GB
        > emit:addr_place   cap=2,000
          > targets:addr_place   cap=2,000
          < targets:addr_place   1.8s  3,957,196 rows (2.21M/s)  cap=2,000  rss 2.66GB  peak 3.12GB  avail 7.93GB  wrote 0.00GB  artefacts +0.00GB
          > targets-join:addr_place
          < targets-join:addr_place   1.1s  9,529,554 rows (8.32M/s)  resident_mb=496.800  buckets_kept=3,957,161  buckets_dropped=35  rss 3.96GB  peak 3.96GB avail 6.64GB  wrote 0.00GB  artefacts +0.00GB
    addr_place     targets    9,529,554 rows   35/3,957,196 buckets over cap 2,000 dropped
          emit addr_place: 28,087,178/28,087,178 pairs  100.0%  2.21M/s  el 12.7s  eta 0.0s  rss 3.36GB  peak 6.41GB  avail 7.26GB  wrote 6.82GB              
        - addr_place: emitted   pairs=28,087,178  bucket_files=32  parquet_mb=158.200  running_total=1,119,438,844
    addr_place       28,087,178 raw pairs -> 32 bucket files   (running total 1,119,438,844)
        < emit:addr_place   16.2s  cap=2,000  files=32  running_pairs=1,119,438,844  rss 3.36GB  peak 3.36GB  avail 7.26GB  wrote 0.15GB  artefacts +0.15GB
        keys: 26/26 keys  100.0%  0/s  el 8m41s  eta 0.0s  rss 3.36GB  peak 6.41GB  avail 7.26GB  wrote 6.82GB                     
      < emit   8m41s  1,119,438,844 rows (2.15M/s)  keys=26  rss 3.36GB  peak 6.41GB  avail 7.26GB  wrote 6.82GB  artefacts +6.82GB
  merging:
      > merge   budget=60  band_min_hits=1  buckets=32
        merge: 32/32 buckets  100.0%  0/s  el 5m31s  eta 0.0s  rss 3.72GB  peak 8.63GB  avail 7.61GB  wrote 7.56GB                                   
      < merge   5m32s  102,937,110 rows (309.8k/s)  budget=60  band_min_hits=1  buckets=32  raw_pairs=687,183,055  candidates_mb=761.800  rss 0.38GB  peak 8.63GB  avail 10.97GB  wrote 0.74GB  artefacts +0.04GB
    - blocking: disk footprint   partial_files=832  partials_gb=6.821  candidates_gb=0.744
  687,183,055 distinct pairs -> 102,937,110 after budget 60 -> test_candidates.parquet (854.4s)   mean 15.0% kept
    < blocking   14m14s  split=test  budget=60  exact_keys=10  bands=16  rss 0.38GB  peak 8.63GB  avail 10.97GB  wrote 7.56GB  artefacts +0.99GB
[telemetry] run 20260927T214912-11452 finished in 14m14s   peak rss 8.85GB   disk written 7.56GB   work+output now 30.19GB
```

## Test Set Featurization

```bash
>> python -m src.features  --split test --merge-buckets 32 --progress

[telemetry] telemetry on  run=20260927T220328-8220  ram 15.7GB  avail 11.35GB  work+output 30.19GB  -> D:\Coding\Projects\working-on\amazon-ml\work\timings.jsonl
features [test]  work=D:\Coding\Projects\working-on\amazon-ml\work
    > features   buckets=32
      > attributes
      < attributes   4.2s  11,702,133 rows (2.79M/s)  s1_mb=310.100  tgt_mb=1,832.400  rss 6.35GB  peak 6.35GB  avail 5.04GB  wrote 0.00GB  artefacts +0.00GB
  attributes: 1,732,544 source-1, 9,969,589 target
      featurize: 102,937,110/102,937,110 pairs  100.0%  200.9k/s  el 8m32s  eta 0.0s  rss 7.38GB  peak 10.12GB  avail 3.78GB  wrote 2.63GB                   
    - features: written   pairs=102,937,110  positives=0  parquet_gb=2.632  bytes_per_row=27.500
  102,937,110 pairs -> test_features.parquet   (517.2s)
    < features   8m37s  buckets=32  rss 4.83GB  peak 10.12GB  avail 6.31GB  wrote 2.63GB  artefacts -0.19GB
[telemetry] run 20260927T220328-8220 finished in 8m37s   peak rss 10.21GB   disk written 2.63GB   work+output now 30.00GB
```

## Test Set Model Prediction

```bash
>> python -m src.model     --split test --folds 5 --coverage 0.8681 --progress

[telemetry] telemetry on  run=20260927T221206-13972  ram 15.7GB  avail 11.08GB  work+output 31.75GB  -> D:\Coding\Projects\working-on\amazon-ml\work\timings.jsonl
model [test]  work=D:\Coding\Projects\working-on\amazon-ml\work
    > model   split=test  folds=5  rounds=600
  features: 102,937,110 pairs, 58 columns
    - feature table   pairs=102,937,110  columns=58  parquet_gb=2.632
  loaded 3 boosters
  ensemble prediction: 102,937,110 rows x 3 booster(s) — expect roughly 1h23m at the measured 61.5k rows/s per booster
    - ensemble prediction planned   rows=102,937,110  boosters=3  est_seconds=5,021
      > predict-ensemble   boosters=3
        ensemble x3: 102,937,110/102,937,110 rows  100.0%  52.1k/s  el 32m57s  eta 0.0s  rss 7.92GB  peak 8.70GB  avail 3.57GB  wrote 0.00GB
      < predict-ensemble   32m57s  102,937,110 rows (52.1k/s)  boosters=3  rss 7.48GB  peak 8.70GB  avail 4.02GB  wrote 0.00GB  artefacts +0.00GB
      > select-matches   topk=8  coverage=0.868
  selection: 1,732,542 entities, 111,685 (6.45%) predict nothing, 5,014,244 predictions, mean 2.89 per entity (coverage 0.8681, topk 8)
      < select-matches   34.0s  5,014,244 rows (147.6k/s)  topk=8  coverage=0.868  rss 7.83GB  peak 8.97GB  avail 3.67GB  wrote 0.00GB  artefacts +0.00GB
    < model   33m33s  5,014,244 rows (2.5k/s)  split=test  folds=5  rounds=600  rss 7.82GB  peak 8.97GB  avail 3.67GB  wrote 0.00GB  artefacts +0.00GB
    > write/matching_results.tsv
    < write/matching_results.tsv   0.1s  1,732,544 rows (30.43M/s)  tsv_mb=83.000  rss 4.67GB  peak 4.67GB  avail 6.71GB  wrote 0.08GB  artefacts +0.01GB
  wrote 1,732,544 rows -> D:\Coding\Projects\working-on\amazon-ml\output\matching_results.tsv
    > write/candidate_pairs.tsv   pairs=102,937,110  buckets=32
      candidate tsv: 32/32 buckets  100.0%  4/s  el 8.0s  eta 0.0s  rss 6.40GB  peak 8.97GB  avail 5.00GB  wrote 0.08GB                    
    < write/candidate_pairs.tsv   10.0s  1,732,544 rows (173.7k/s)  pairs=102,937,110  buckets=32  grouped_rows=1,732,544  tsv_mb=1,286.600  rss 8.57GB  peak 8.57GB  avail 2.48GB  wrote 1.26GB  artefacts -0.34GB
  wrote 1,732,544 rows -> D:\Coding\Projects\working-on\amazon-ml\output\candidate_pairs.tsv
  done in 2026.9s
[telemetry] run 20260927T221206-13972 finished in 33m46s   peak rss 8.97GB   disk written 1.34GB   work+output now 31.41GB
```
