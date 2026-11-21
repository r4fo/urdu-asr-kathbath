# MFCC + GMM-HMM baseline (Member 3)

A classical speech recognizer built with Kaldi: MFCC features, Gaussian-mixture HMMs on letter units
(a grapheme lexicon, so no pronunciation dictionary is needed), and a 3-gram word language model.
It is small and trained for about an hour on CPU. It exists to show where the classical method stands
next to the neural systems, not to get a strong number.

| Item | Setting |
| --- | --- |
| Toolkit | Kaldi (revision: TODO, from `results/timing/gmm_hmm.csv`) |
| Features | 13 MFCC (40 mel bins, 16 kHz), per-speaker CMVN, later with deltas, then LDA + MLLT (splice ±3) |
| Units | One unit per letter of the normalized transcript (40 letters), plus `sil` and `spn` |
| Lexicon | Words of the **train split only**, each spelled with its letters (`make_dict.py`), `<unk>` maps to `spn` |
| Training | monophone → tri1 (deltas, 2000 leaves, 10000 Gaussians) → tri2 (LDA+MLLT, 2500 leaves, 15000 Gaussians) |
| Language model | 3-gram, KenLM `lmplz --discount_fallback`, **train split text only** (11,868 sentences) |
| Decoding | Lattice decoding, beam 13; LM weight 7-17 and word insertion penalty 0 / 0.5 / 1 chosen **on dev** |
| Test | Decoded once, after the dev settings were frozen |
| Seed / randomness | Kaldi training is deterministic for the same data and the same number of jobs |

## Files

| File | What it does |
| --- | --- |
| `../../scripts/make_kaldi_dirs.py` | Split CSVs → Kaldi data folders (`wav.scp`, `text`, `utt2spk`, `spk2utt`); the text is normalized with `src/normalize.py`; stops if a speaker is in two splits |
| `make_dict.py` | Train text → grapheme lexicon and phone lists; prints the OOV rate of dev and test |
| `run.sh` | All stages: data, lexicon, LM, MFCC, monophone, tri1, tri2, dev decoding, and (with `--test`) the single test decoding |
| `to_predictions.py` | Kaldi hypotheses → `results/predictions/gmm_hmm.csv` (`utt_id,hyp`, raw output, not normalized) |
| `make_timing.py` | Decoding time → `results/timing/gmm_hmm.csv` |
| `work/` | Kaldi folders, features and models. Created by `run.sh`, ignored by git |

Tests: `tests/test_gmm_hmm.py` (no Kaldi needed).

## How to run

Kaldi runs on Linux, WSL2 or Docker, not plain Windows. With Docker (the repo is mounted at `/repo`):

```
docker run -it -v <path to the repo>:/repo kaldiasr/kaldi:latest bash
cd /repo
bash baselines/gmm_hmm/run.sh            # train + decode dev, writes results/predictions/gmm_hmm_dev.csv
python3 src/evaluate.py --split data/splits/dev.csv --pred results/predictions/gmm_hmm_dev.csv
bash baselines/gmm_hmm/run.sh --test     # once, after the dev settings are frozen
python3 src/evaluate.py --split data/splits/test.csv --pred results/predictions/gmm_hmm.csv
python3 src/evaluate.py --split data/splits/test.csv --pred results/predictions/gmm_hmm.csv \
    --exclude data/splits/test_overlap.csv
```

## Vocabulary limit (measured)

The lexicon only has the 14,470 words of the train text, so a word that is in dev or test but not in train
cannot be recognized. The OOV rate (share of running words that are not in the vocabulary) is **5.24% on dev**
and **5.52% on test** (`make_dict.py` output). Together with the letter-by-letter acoustic model this sets a
floor under the WER of this system that the end-to-end character models do not have.

## Results

TODO: filled from `results/results.csv` after the runs, never by hand (see `scripts/make_tables.py`).

## Known problems

TODO: write here anything that did not work (Kaldi installation, memory, time), with the exact error.
