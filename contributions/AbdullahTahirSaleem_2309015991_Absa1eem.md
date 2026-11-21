# Abdullah Tahir Saleem · 2309015991 · Absa1eem

## Role

Member 3: classical baseline, MFCC + GMM-HMM with Kaldi (`baselines/gmm_hmm/`).

## What I did

- Wrote `scripts/make_kaldi_dirs.py`, which turns the shared split CSVs into Kaldi data folders using the
  team's `src/normalize.py` and stops if a speaker is in two splits.
- Wrote `baselines/gmm_hmm/` (`make_dict.py`, `run.sh`, `to_predictions.py`, `make_timing.py`) and
  `tests/test_gmm_hmm.py`.
- Built the grapheme lexicon and a 3-gram LM from the train split only, trained monophone, tri1 and tri2
  models, chose the LM weight on dev, and decoded the test set once.
- Wrote the GMM-HMM section of the README.

## Results

See `results/results.csv` (system `gmm_hmm`) and `results/timing/gmm_hmm.csv`.

## How I used Claude

Claude (Anthropic's AI assistant) helped me write the scripts above and explained the Kaldi steps. I ran them,
checked the outputs and can explain each step.
