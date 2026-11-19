# Urdu ASR Benchmark on Kathbath (Team 7, COE025 Project 3)

> Status: work in progress. Every section marked TODO is filled before the deadline.
> Every number in this README is produced by committed code (`scripts/make_tables.py`).

## 1. Overview

We build and compare automatic speech recognition (ASR) systems for **Urdu**.
We run every course baseline that supports Urdu, fine-tune Whisper on our own training
split as our contribution, and evaluate all systems on the same speaker-disjoint test set
with the same text normalization and the same scoring (WER and CER).

## 2. Team

| Member | Role | GitHub |
| --- | --- | --- |
| TODO | Data, Whisper fine-tuning, repository | TODO |
| Muhammed Isa Loonat | Splits and evaluation | [isaloonat](https://github.com/isaloonat) |
| TODO | GMM-HMM baseline | TODO |
| TODO | CTC, attention, RNN-T baselines | TODO |
| TODO | Pretrained checkpoints, metrics, presentation | TODO |

Each member's contribution is described in `contributions/`.

## 3. Dataset

- **Source:** Kathbath, Urdu subset, by AI4Bharat (https://huggingface.co/datasets/ai4bharat/Kathbath).
  Read speech, human-verified, with a speaker ID for every utterance.
- **License:** the dataset card licenses the packaging under CC0 1.0, and its metadata lists
  CC BY 4.0. We follow the stricter one (CC BY 4.0) and credit AI4Bharat. The read sentences
  come from IndicCorp; AI4Bharat states that it does not own that raw text.
- **Why Urdu:** TODO (no other group, under-resourced for speech, with a citation).
- **What we use:** TODO hours from the clean part, plus the official Test-Unknown split.

Citation:

```
@article{javed2022indicsuperb,
  title   = {IndicSUPERB: A Speech Processing Universal Performance Benchmark for Indian Languages},
  author  = {Javed, Tahir and Bhogale, Kaushal Santosh and Raman, Abhigyan and Kunchukuttan, Anoop and Kumar, Pratyush and Khapra, Mitesh M.},
  journal = {arXiv preprint arXiv:2208.11761},
  year    = {2022}
}
```

## 4. Cleaning

`scripts/clean.py` turns `data/raw/` into `data/clean/`. Every removed or changed utterance is
logged with its reason in `data/clean/cleaning_log.csv`.

| Problem | How we detect it | Utterances affected |
| --- | --- | --- |
| TODO | TODO | TODO |

## 5. Splits

The Hugging Face version of Kathbath has no Test-Unknown split, and every valid speaker is also a train
speaker. So `scripts/split.py` splits all 47 speakers ourselves, **by speaker** (seed `42`): every
utterance of a speaker goes to the same split, and no speaker is in two splits.

- Test: 9 speakers (4 F / 5 M), dev: 6 speakers (3 F / 3 M), the rest train. Genders are divided in
  proportion to the speaker counts (21 F / 26 M).
- Speakers with fewer than 50 clips (9 speakers, 1 to 37 clips) always go to train. In test they would
  add a speaker but almost no audio.
- Speakers are drawn at random with seed 42; the first draw where test has 17-19% and dev 10-12% of the
  hours is kept (draw 586). This stops a lucky draw of very large or very small speakers from skewing the sizes.

| Split | Speakers (F / M) | Utterances | Hours |
| --- | --- | --- | --- |
| Train | 32 (14 / 18) | 11,868 | 17.64 (70.6%) |
| Dev | 6 (3 / 3) | 1,956 | 2.72 (10.9%) |
| Test | 9 (4 / 5) | 3,031 | 4.62 (18.5%) |

Test speakers: 15, 216, 295, 330, 617, 835, 842, 1064, 1094. Dev speakers: 264, 318, 485, 627, 841, 1049.
Counts are in `data/splits/split_stats.csv`.

Test sentences that also appear in train (after normalization): **1 of 3,031**, the one sentence read twice
in the data (speaker 68 in train, speaker 842 in test, flagged in `data/clean/cleaning_log.csv`). It is listed in
`data/splits/test_overlap.csv`. For systems with a language model we also report WER without it.

Dev is used to choose settings and checkpoints. Test is used once, at the end, after all settings are frozen.

## 6. Metrics

TODO: why WER and CER fit Urdu ASR, with the Hugging Face audio course, chapter 5
(https://huggingface.co/learn/audio-course/en/chapter5/evaluation).

**Normalization policy (fixed on 19 November 2026, before any test results; not changed after).**
`src/evaluate.py` applies `src/normalize.py` to references and predictions in exactly the same way.
Nobody normalizes their own predictions. In order:

1. Numbers written with digits become Urdu words, as the references write them (`src/urdu_numbers.py`):
   "2015" → "دو ہزار پندرہ", "10,000" → "دس ہزار". The references contain no digits, so this only changes
   predictions, the same way for every system.
2. Unicode NFC; invisible characters (zero-width, direction marks) and tatweel removed.
3. Letter variants unified to the forms the references use: ك → ک, ي / ى → ی, ه → ہ,
   آ / أ / إ → ا, ؤ → و, ۂ / ۀ → ہ, ۓ → ے. ئ, ء, ے and ھ are kept.
4. Diacritics removed (U+064B to U+0655, U+0670).
5. Punctuation (۔ ، ؟ ؛ and others) becomes a space; spaces are collapsed.

WER and CER are corpus-level (total errors / total reference words or characters; CER counts spaces).
The 95% confidence interval on WER comes from 1,000 bootstrap resamples of test utterances (seed 42).

The main results table uses this full normalization for every system. A small extra table shows how much
normalization matters, for zero-shot Whisper-small, Whisper-large-v3 and our fine-tuned model:
(1) no normalization (`--no-normalize`), (2) letters and punctuation only (`--no-digits`),
(3) full normalization (the main results).

## 7. Baselines

Baselines are models we did not train further, plus standard methods we implemented and trained ourselves.

| System | Exact checkpoint / config | Revision | Trained by us? |
| --- | --- | --- | --- |
| MFCC + GMM-HMM | `baselines/gmm_hmm/` | - | yes |
| CTC encoder | `baselines/ctc_attention_rnnt/` | - | yes |
| Hybrid CTC/attention | `baselines/ctc_attention_rnnt/` | - | yes |
| RNN-T | `baselines/ctc_attention_rnnt/` | - | yes |
| Whisper (zero-shot) | `openai/whisper-small` | TODO | no |
| wav2vec 2.0 + CTC | TODO | TODO | no |
| HuBERT + CTC | TODO | TODO | no |

Models that do not support Urdu (not run):

| Model | Reason | Model card |
| --- | --- | --- |
| `nvidia/canary-1b-v2` | 25 European languages only | https://huggingface.co/nvidia/canary-1b-v2 |
| `nvidia/parakeet-tdt-0.6b-v3` | 25 European languages only | https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 |
| `Qwen/Qwen3-ASR-1.7B`, `Qwen/Qwen3-ASR-0.6B` | Urdu not in the supported language list | https://huggingface.co/Qwen/Qwen3-ASR-1.7B |

## 8. Our contribution

TODO: fine-tuning `openai/whisper-small` on our train split (settings, seed, checkpoint choice on dev).

## 9. Results

Generated by `scripts/make_tables.py` from `results/results.csv`. Same test clips, same references,
same normalization for every system.

Baselines: TODO

Our models: TODO

## 10. How to reproduce

```
pip install -r requirements.txt
python scripts/clean.py
python scripts/split.py
# train / run each system (see its folder)
python src/evaluate.py ...
python scripts/make_tables.py
pytest tests/
```

Seeds: TODO.

## 11. Repository layout

| Path | Content |
| --- | --- |
| `README.md` | This file |
| `LICENSE` | MIT license for our own code (the dataset has its own license, see section 3) |
| `requirements.txt` | Pinned package versions |
| `.gitignore` | Files git ignores (caches, virtual environments) |
| `contributions/` | One file per member: `<NAME>_<STUDENT_ID>_<GITHUB_USERNAME>.md` |
| `data/raw/` | Kathbath Urdu audio and transcripts as downloaded |
| `data/clean/` | Cleaned data and `cleaning_log.csv` |
| `data/splits/` | Train / dev / test manifests and split statistics |
| `scripts/` | Data cleaning, splitting, and table generation |
| `src/` | Text normalization and WER/CER evaluation, shared by all systems |
| `baselines/gmm_hmm/` | MFCC + GMM-HMM baseline |
| `baselines/ctc_attention_rnnt/` | CTC, hybrid CTC/attention, and RNN-T baselines |
| `baselines/pretrained/` | Scripts that run pretrained checkpoints without training |
| `models/whisper_finetune/` | Our Whisper fine-tuning code |
| `results/predictions/` | One prediction file per system |
| `results/results.csv` | WER and CER of every system |
| `tests/` | Tests for cleaning, splitting, and evaluation |

`.gitkeep` files only keep empty folders in git. They are removed once a folder has real files.
