# MFCC + GMM-HMM baseline (Member 3)

A classical speech recognizer built with Kaldi: MFCC features, Gaussian-mixture HMMs on letter units
(a grapheme lexicon, so no pronunciation dictionary is needed), and a 3-gram word language model.
It is small and trained on CPU. It exists to show where the classical method stands
next to the neural systems, not to get a strong number.

| Item | Setting |
| --- | --- |
| Toolkit | Kaldi, git revision `f400766` (Docker image `kaldiasr/kaldi:latest`), run on CPU only |
| Features | 13 MFCC (40 mel bins, 16 kHz), per-speaker CMVN, later with deltas, then LDA + MLLT (splice ±3) |
| Units | One unit per letter of the normalized transcript (40 letters), plus `sil` and `spn` |
| Lexicon | Words of the **train split only**, each spelled with its letters (`make_dict.py`), `<unk>` maps to `spn` |
| Training | monophone → tri1 (deltas, 2000 leaves, 10000 Gaussians) → tri2 (LDA+MLLT, 2500 leaves, 15000 Gaussians) |
| Language model | 3-gram, KenLM `lmplz --discount_fallback`, **train split text only** (11,868 sentences, 14,470 distinct words) |
| Decoding | Lattice decoding, beam 13; LM weight 7-17 and word insertion penalty 0 / 0.5 / 1 tried **on dev**; chosen: LM weight 15, penalty 0.0 |
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

Scored with `src/evaluate.py` (full normalization, the same for every system). WER and CER are fractions of
reference words and characters; the interval is a bootstrap 95% confidence interval on WER.

| Set | Clips | WER | WER 95% CI | CER |
| --- | --- | --- | --- | --- |
| dev (LM weight and penalty chosen here) | 1,956 | 37.29% | 36.25 to 38.32 | 19.59% |
| test | 3,031 | 39.62% | 38.78 to 40.40 | 19.84% |
| test, only sentences not in train (`--exclude data/splits/test_overlap.csv`) | 3,030 | 39.63% | 38.79 to 40.45 | 19.85% |

The test set was decoded once, after the dev settings were fixed. Test is about 2 points worse than dev, which is
the usual gap after choosing settings on dev. Only one test sentence also appears in train, so removing it changes
nothing (39.62% against 39.63%).

Speed (`results/timing/gmm_hmm.csv`): 16,635.62 s of test audio decoded in 2,525.35 s, **RTF 0.1518**, on a 12th Gen
Intel Core i7-1255U (12 logical processors, no GPU) with 9 parallel decoding jobs. This RTF is for CPU decoding only and
is not comparable to the GPU rows.

What the numbers say: the 5.5% of test words that are not in the train vocabulary (see above) can never be
recognized: each of them is at least one error, so they alone account for at least 5.5 points of the 39.6 points of
WER (1,857 of 33,662 reference words). The rest comes from the simple acoustic model (Gaussian mixtures per letter
state, no neural network) and the small 3-gram LM. CER (19.8%) is about half of
WER, as expected for a system that gets many letters right but misses whole words.

## Known problems

- **Kaldi does not run on Windows.** The model was trained inside Docker Desktop (WSL2 backend) with the image
  `kaldiasr/kaldi:latest`. Docker first stopped with "WSL not installed" until `wsl --install` was run and the computer restarted.
- **Bug found and fixed in our own `make_dict.py`:** it wrote `extra_questions.txt` with one empty line, and Kaldi's
  `validate_dict_dir.pl` rejects empty lines ("empty line in extra_questions.txt"). The file is now written empty.
- **Warning "Tree has pdf-id 1 with no stats; corresponding phone list: 2"** during tri1 and tri2 training. It is
  harmless: phone 2 is `spn` (the unknown-word sound), and the train lexicon has no unknown words, so `spn` has no
  training data.
- **The LM weight chosen on dev (15) is near the top of the range tried (7 to 17).** A larger weight might have scored
  slightly better on dev. The range was fixed before any test result and was not extended afterwards.
- **Vocabulary limit:** 5.24% of dev words and 5.52% of test words are not in the train lexicon (`data/oov_report.txt`
  in `work/`), so they are always wrong.
