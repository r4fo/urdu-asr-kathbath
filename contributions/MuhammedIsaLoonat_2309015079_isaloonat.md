# Muhammed Isa Loonat

- Student ID: 2309015079
- GitHub: [isaloonat](https://github.com/isaloonat)
- Role: Member 2, splits and evaluation

## What I did

- `src/normalize.py`: one `normalize(text)` used by every system on references and predictions
  (digits to Urdu words via `src/urdu_numbers.py`, NFC, Urdu letter variants including آ أ ؤ ۂ ۓ,
  diacritics, punctuation, spaces).
- `src/evaluate.py`: WER and CER with `jiwer`, a bootstrap 95% CI on WER (1000 resamples, seed 42),
  a check that every `utt_id` has a prediction, and one row per run in `results/results.csv`.
- `scripts/split.py` and the frozen split in `data/splits/`: all 47 speakers split by speaker (seed 42),
  test 9 speakers (4 F / 5 M, 4.62 h), dev 6 (3 F / 3 M, 2.72 h), train 32 (17.64 h); writes
  `split_stats.csv` and `test_overlap.csv` (test sentences also in train).
- Tests: `tests/test_normalize.py`, `tests/test_evaluate.py` (hand-checked WER and CER),
  `tests/test_split.py` (no speaker in two splits, every clip in exactly one split).
