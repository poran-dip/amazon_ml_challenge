# Amazon ML Challenge

Folder structure:

```bash
amazon_ml_challenge/
├── student_resource/                       # provided by Amazon, untouched, gitignored
│   ├── dataset/
│   │   ├── train/
│   │   │   ├── train_source1.tsv           # 2,206,821 rows
│   │   │   ├── train_source2.tsv           # 5,034,616 rows
│   │   │   ├── train_source3.tsv           # 5,285,603 rows
│   │   │   └── train_ground_truth.tsv      # 2,206,821 rows
│   │   └── test/
│   │       ├── test_source1.tsv            # 1,732,544 rows
│   │       ├── test_source2.tsv            # 4,887,273 rows
│   │       └── test_source3.tsv            # 5,082,316 rows
│   ├── utils/
│   │   └── validate_submission.py          # format validator
│   ├── Documentation_template.md           # methodology template
│   └── README.md                           # spec
│
├── code/
│   └── business_entity_resolution/
│       ├── venv/
│       ├── src/                            # empty for now
│       ├── README.md                       # submission README, empty for now
│       └── requirements.txt
│
├── Documentation_template.md               # working copy, filled over time
├── README.md                               # repo README (this one, shows on GitHub)
└── .gitignore
```

Final submission format:

```bash
pyroflies_submission.zip
├── output/
│   ├── matching_results.tsv        # final matches
│   └── candidate_pairs.tsv         # blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # source code
│       ├── README.md               # how to reproduce end-to-end
│       └── requirements.txt        # pinned dependencies / environment
└── Documentation_template.md       # methodology write-up (filled-in template)
```
