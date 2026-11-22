# Urdu ASR Benchmark on Kathbath (Team 7, COE025 Project 3)

Every number in this README comes from committed code: result tables from `scripts/make_tables.py`
(output `results/tables.md`), error analysis from `scripts/error_analysis.py` (output `results/error_analysis/`).

## 1. Overview

We build and compare automatic speech recognition (ASR) systems for **Urdu**: 13 systems, all scored on the same
speaker-disjoint test set with the same text normalization and the same scoring (WER and CER).

- **Course baselines that support Urdu**, trained by us: Kaldi MFCC + GMM-HMM, and ESPnet CTC, hybrid
  CTC/attention and RNN-T.
- **Published checkpoints that support Urdu**, run as they are (vanilla): Whisper-small and Whisper-large-v3
  zero-shot, and two wav2vec 2.0 models fine-tuned on Common Voice Urdu by others (one with and without its LM).
- **Our contribution:** Whisper-small and wav2vec 2.0 XLS-R 300M fine-tuned on our training split, and a KenLM
  3-gram built from training text only, with its weights chosen on dev.

**Main result:** our wav2vec 2.0 XLS-R + KenLM system has the lowest test WER, **18.29%** (95% CI 17.75 to 18.86),
ahead of our fine-tuned Whisper-small (19.74%) and Whisper-large-v3 zero-shot (21.64%), and it has the lowest
measured RTF of the four best systems (0.0043, about 13 times lower than Whisper-small with beam 5; see 9.1 on hardware).

## 2. Team

| Member | Role | GitHub |
| --- | --- | --- |
| Rafik Ajam | Data download, cleaning and checks; Whisper and wav2vec 2.0 fine-tuning; KenLM; pretrained checkpoints; repository and coordination | [r4fo](https://github.com/r4fo) |
| Muhammed Isa Loonat | Splits, normalization and evaluation | [isaloonat](https://github.com/isaloonat) |
| Abdullah Tahir Saleem | Kaldi GMM-HMM baseline | [Absa1eem](https://github.com/Absa1eem) |
| Amro Akel | ESPnet CTC, hybrid CTC/attention and RNN-T baselines | [AmroAke1](https://github.com/AmroAke1) |
| Mohamed Aziz Ben Thabet | Project review, Q&A preparation, presentation | [azizbth](https://github.com/azizbth) |

Each member's contribution is described in `contributions/`.

## 3. Dataset

- **Source:** Kathbath, Urdu subset, by AI4Bharat (https://huggingface.co/datasets/ai4bharat/Kathbath).
  Read speech, human-verified, with a speaker ID for every utterance, 16 kHz audio.
- **License:** the dataset card licenses the packaging under CC0 1.0, and its metadata lists
  CC BY 4.0. We follow the stricter one (CC BY 4.0) and credit AI4Bharat. The read sentences
  come from IndicCorp; AI4Bharat states that it does not own that raw text.
- **Why Urdu:** no other group in the course chose it, open human-verified Urdu speech data is scarce, and among
  the course's pretrained models only Whisper supports it (the Urdu wav2vec 2.0 models we ran are community checkpoints), so most of the work is in fine-tuning and in
  baselines trained from scratch. The Perso-Arabic script also has letters that sound the same (section 9.3).
- **What we use:** a 25-hour subset of the Hugging Face `train` and `valid` splits (the Hugging Face version has no
  Test-Unknown split). `scripts/download_kathbath_urdu.py` caps the hours per speaker (found by binary search)
  so no voice dominates, samples the rest with seed 42, and stores the audio as 16-bit FLAC in `data/raw/`
  (16,856 utterances, 47 speakers). We then split by speaker ourselves (section 5).

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

`scripts/clean.py` turns `data/raw/` into `data/clean/metadata.csv` (paths still point into `data/raw/`). Every
flagged or removed utterance is logged with its reason in `data/clean/cleaning_log.csv`, totals in
`data/clean/cleaning_summary.csv`. Tests: `tests/test_clean.py`.

| Problem | How we detect it | Utterances affected |
| --- | --- | --- |
| Unreadable audio | `soundfile` cannot read the file | 0 |
| Duplicate audio | identical MD5 hash | 0 |
| Wrong duration in the metadata | file length differs from the `duration` column | 0 |
| Wrong sample rate | not 16 kHz | 0 |
| Too short or too long | under 0.5 s or over 30 s | 0 |
| Silence | level under -50 dBFS | 0 |
| Clipping | more than 1% of samples at full scale | 0 |
| Empty or Latin text | no text, or Latin letters in an Urdu transcript | 0 |
| Speaking rate out of range | under 3 or over 30 characters per second (text and audio do not match) | 0 |
| Same sentence read twice | identical normalized text | 2 flagged, kept (one in train, one in test, see section 5) |
| Odd audio found by profiling | `scripts/profile_data.py`: speech activity and level; 60 candidates in `data/clean/profile_candidates.csv`, the 10 most extreme listened to (`data/clean/manual_review.csv`) | **1 removed** (`844424930857826-622-m`), 9 kept |
| Wrong transcript | `scripts/find_label_errors.py`: Whisper-small zero-shot vs the transcript; the 30 biggest mismatches in `data/clean/label_check_candidates.csv` reviewed | 0 changed |

Result: 16,856 → **16,855** utterances. Kathbath is human-verified, so the automatic checks found almost
nothing; we report that instead of inventing problems.

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

Following the Hugging Face audio course, chapter 5 (https://huggingface.co/learn/audio-course/en/chapter5/evaluation):

- **WER** (word error rate) is the main metric: substitutions, deletions and insertions over the number of
  reference words. It is the standard ASR metric and matches what a reader sees: a word is right or wrong.
- **CER** (character error rate) is reported too. Urdu has letters that sound the same and spelling variants, so a
  word with one wrong letter is a full error in WER; CER shows how close such outputs are. It also makes the
  word-based GMM-HMM, the character-based end-to-end models and Whisper's tokens comparable.
- **Normalization:** the course recommends scoring normalized text so that formatting (punctuation, number
  format) is not counted as a recognition error, and showing how much normalization changes the numbers. We do
  both (policy below, effect in section 9).
- **RTF** (real-time factor) = decoding time / audio duration, measured after a warm-up batch with
  `torch.cuda.synchronize()`, without reading audio from disk. It is only comparable on the same hardware, so every
  row states its hardware.

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
normalization matters, for zero-shot Whisper-small and our fine-tuned Whisper-small:
(1) no normalization (`--no-normalize`), (2) letters and punctuation only (`--no-digits`),
(3) full normalization (the main results).

## 7. Baselines

| System | Exact checkpoint / config | Revision | Trained by us? |
| --- | --- | --- | --- |
| MFCC + GMM-HMM (Kaldi, tri2 LDA+MLLT) | `baselines/gmm_hmm/` | code at commit `f400766` | yes |
| CTC encoder (ESPnet 202511) | `baselines/ctc_attention_rnnt/conf/train_ctc.yaml` | code at commit `17d92a46` | yes |
| Hybrid CTC/attention (ESPnet 202511) | `baselines/ctc_attention_rnnt/conf/train_hybrid.yaml` | code at commit `17d92a46` | yes |
| RNN-T (ESPnet 202511) | `baselines/ctc_attention_rnnt/conf/train_rnnt.yaml` | code at commit `17d92a46` | yes |
| Whisper-small, zero-shot | `openai/whisper-small` | `973afd24965f72e36ca33b3055d56a652f456b4d` | no |
| Whisper-large-v3, zero-shot | `openai/whisper-large-v3` | `06f233fe06e710322aca913c1bc4249a0d71fce1` | no |
| wav2vec 2.0 + CTC, greedy and with its own KenLM | `anuragshas/wav2vec2-xls-r-300m-ur-cv9-with-lm` | `0fee38baf834b841d27923ac9c09676652963237` | no |
| wav2vec 2.0 + CTC, greedy | `kingabzpro/wav2vec2-urdu` | `627818e6c8218012d468cdc82e7f9b08c6f8ac13` | no |

Whisper zero-shot is decoded with `language="urdu"`, `task="transcribe"`, beam 1. The two wav2vec 2.0 checkpoints use
their own processor; anuragshas with its LM uses the LM and weights it ships with, pyctcdecode's default beam (100),
on Colab (`baselines/pretrained/anuragshas_lm_colab.ipynb`), since `kenlm` does not install on Windows. The GMM-HMM and
ESPnet details are in their folders' READMEs.

Models on the course list that do not support Urdu (not run):

| Model | Reason | Model card |
| --- | --- | --- |
| `nvidia/canary-1b-v2` | 25 European languages only | https://huggingface.co/nvidia/canary-1b-v2 |
| `nvidia/parakeet-tdt-0.6b-v3` | 25 European languages only | https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 |
| `Qwen/Qwen3-ASR-1.7B`, `Qwen/Qwen3-ASR-0.6B` | Urdu not in the supported language list | https://huggingface.co/Qwen/Qwen3-ASR-1.7B |
| HuBERT + CTC (`facebook/hubert-large-ls960-ft` and the other `facebook/hubert-*` checkpoints) | pretrained on English audio only (LibriLight), CTC heads fine-tuned on English; no Urdu HuBERT checkpoint exists, so the vanilla model cannot output Urdu | https://huggingface.co/facebook/hubert-large-ls960-ft |

## 8. Our contribution

All three systems are tuned on dev only; test was decoded once per system, after it was frozen.

**Whisper-small fine-tuned** (`models/whisper_finetune/`). `train.py` with its defaults: `openai/whisper-small`
(revision above), learning rate 1e-5, 200 warm-up steps, batch 8 × gradient accumulation 2, fp16, gradient
checkpointing, AdamW, seed 42, evaluated every 250 steps on the first 300 dev clips; the checkpoint with the lowest
dev WER is kept. Decoding: beam 5 is the main system (dev WER 17.36% against 17.94% for beam 1); beam 1 is reported
as the faster option. Frozen on 19 November 2026.
Weights: [`r4fo/whisper-small-urdu-kathbath`](https://huggingface.co/r4fo/whisper-small-urdu-kathbath),
revision `b060bc34ee1c739c3ec96049c5f26f245e0edf68`.

**wav2vec 2.0 XLS-R 300M fine-tuned** (`models/wav2vec2_finetune/`). `facebook/wav2vec2-xls-r-300m` with a new CTC
head over the characters of the train transcripts, 15 epochs (`--epochs 15`, other settings the `train.py`
defaults: learning rate 3e-4, 500 warm-up steps, batch 8 × 2, frozen feature encoder, layerdrop 0.05, time masking
0.05, length-grouped batches, seed 42), checkpoint with the lowest dev WER.
Weights: [`r4fo/wav2vec2-xls-r-300m-urdu-kathbath`](https://huggingface.co/r4fo/wav2vec2-xls-r-300m-urdu-kathbath),
revision `da12c3a661de35b65275157b978ead8405640da7`. Timing files that say `revision local` were run from the local
copy of these same weights.

**KenLM 3-gram** (`models/wav2vec2_finetune/build_lm.sh`, `tune_lm.py`, `kenlm_colab.ipynb`). Built with
`lmplz -o 3` from the **train split text only** (`lm/train_3gram.arpa`); dev and test text are never used. Beam
search with pyctcdecode, beam 50. `alpha` (LM weight) and `beta` (word bonus) were chosen on dev from a 4 × 3 grid
(`results/lm_tuning_dev.csv`): alpha 0.7, beta 2.0. Run on Colab because `kenlm` does not install on Windows.

## 9. Results

### 9.1 Test set

| # | System | Type | WER [95% CI] | CER | WER without the 1 overlapping sentence | RTF | Hardware |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | wav2vec2 XLS-R 300M fine-tuned + KenLM 3-gram | our fine-tuned system | 18.29% [17.75%, 18.86%] | 5.59% | 18.30% | 0.0043 | Tesla T4 |
| 2 | Whisper-small fine-tuned, beam 5 | our fine-tuned system | 19.74% [19.19%, 20.33%] | 6.01% | 19.74% | 0.0542 | NVIDIA GeForce RTX 4060 |
| 3 | Whisper-small fine-tuned, beam 1 | our fine-tuned system | 20.39% [19.80%, 20.98%] | 6.17% | 20.39% | 0.0081 | NVIDIA GeForce RTX 4060 |
| 4 | Whisper-large-v3, zero-shot | published checkpoint, run as is | 21.64% [21.06%, 22.27%] | 6.63% | 21.64% | 0.0403 | NVIDIA GeForce RTX 4060 |
| 5 | wav2vec2 XLS-R 300M fine-tuned, greedy | our fine-tuned system | 23.28% [22.63%, 23.86%] | 6.63% | 23.28% | 0.0023 | NVIDIA GeForce RTX 4060 |
| 6 | anuragshas XLS-R Urdu + its own LM | published checkpoint, run as is | 27.73% [27.02%, 28.37%] | 8.96% | 27.74% | 0.0074 | Tesla T4 |
| 7 | ESPnet RNN-T | course baseline, trained by us | 34.34% [33.57%, 35.05%] | 13.07% | 34.34% | 0.3050 | Tesla T4 |
| 8 | anuragshas XLS-R Urdu, greedy | published checkpoint, run as is | 39.01% [38.30%, 39.73%] | 11.52% | 39.01% | 0.0021 | NVIDIA GeForce RTX 4060 |
| 9 | Kaldi MFCC + GMM-HMM | course baseline, trained by us | 39.62% [38.78%, 40.40%] | 19.84% | 39.63% | 0.1518 | CPU only, 12th Gen Intel(R) Core(TM) i7-1255U, 12 cores |
| 10 | ESPnet hybrid CTC/attention | course baseline, trained by us | 40.71% [39.92%, 41.45%] | 13.88% | 40.71% | 0.4039 | Tesla T4 |
| 11 | Whisper-small, zero-shot | published checkpoint, run as is | 42.66% [40.97%, 44.64%] | 15.80% | 42.66% | 0.0102 | NVIDIA GeForce RTX 4060 |
| 12 | ESPnet CTC | course baseline, trained by us | 44.60% [43.77%, 45.38%] | 14.64% | 44.60% | 0.2402 | Tesla T4 |
| 13 | kingabzpro wav2vec2 Urdu, greedy | published checkpoint, run as is | 52.34% [51.56%, 53.11%] | 19.00% | 52.33% | 0.0008 | NVIDIA GeForce RTX 4060 |

The same greedy wav2vec2 model run again on Colab (Tesla T4), only to time greedy and LM decoding on the same GPU: WER 23.28%, RTF 0.0030.

The greedy wav2vec 2.0 model was also run on Colab (Tesla T4) only to time greedy and LM decoding on the same GPU:
WER 23.28% (the same model on the RTX 4060 gives 23.28% too), RTF 0.0030 against 0.0043 with the LM.

What the table says:

- **Fine-tuning works.** Whisper-small goes from 42.66% zero-shot to 19.74% after fine-tuning on 17.6 hours, and
  beats Whisper-large-v3 zero-shot (21.64%), a model about six times larger. The margin is about 2 points, and the
  confidence intervals do not overlap.
- **The language model is the biggest single gain.** KenLM takes wav2vec 2.0 from 23.28% to 18.29%. It is not
  leakage: the LM uses train text only, and without the one test sentence that also appears in train the WER is
  18.30%.
- **The most accurate system is also cheap.** wav2vec 2.0 + KenLM decodes at RTF 0.0043 on a T4, Whisper-small
  beam 5 at 0.0542 on an RTX 4060: about 13 times lower. The GPUs differ, but the T4 is the slower one (the same
  greedy wav2vec 2.0 model: RTF 0.0030 on the T4, 0.0023 on the RTX 4060), so the gap is not an effect of better hardware.
- **Pretraining matters at this data size.** The ESPnet models trained from scratch reach 34 to 45%; every
  pretrained model fine-tuned on our data is under 24%.
- **Published Urdu checkpoints transfer poorly** to Kathbath (39.01% and 52.34% greedy). They were fine-tuned on a
  few hours of Common Voice Urdu (other speakers, other text). anuragshas's own LM helps a lot (39.01% → 27.73%),
  the same effect as ours.
- **Test is harder than dev** by about 2 to 3 points for every system (for example Whisper fine-tuned 17.36% → 19.74%,
  wav2vec 2.0 greedy 20.84% → 23.28%). The gap is the same for all systems, which points to harder test speakers
  rather than overfitting to dev.

### 9.2 Dev set and normalization

Dev results (used for all choices):

| System | WER [95% CI] | CER |
| --- | --- | --- |
| `whisper_small_ft_dev_beam1` | 17.94% [17.28%, 18.62%] | 5.44% |
| `whisper_small_ft_dev_beam5` | 17.36% [16.70%, 18.07%] | 5.21% |
| `whisper_small_zeroshot` | 35.85% [34.77%, 37.03%] | 12.18% |
| `w2v2_xlsr_ft_dev_greedy` | 20.84% [20.05%, 21.62%] | 5.79% |
| `gmm_hmm_dev` | 37.29% [36.25%, 38.32%] | 19.59% |
| `ctc_dev` | 40.28% [39.25%, 41.19%] | 13.05% |
| `hybrid_dev` | 36.04% [35.12%, 36.86%] | 12.03% |
| `rnnt_dev` | 30.07% [29.11%, 30.95%] | 11.32% |

How much the normalization changes test WER (section 6):

| System | No normalization | Letters and punctuation only | Full (main table) |
| --- | --- | --- | --- |
| Whisper-small fine-tuned, beam 5 | 19.74% | 19.74% | 19.74% |
| Whisper-small, zero-shot | 43.21% | 42.80% | 42.66% |

Normalization moves zero-shot Whisper by 0.55 points and our fine-tuned model by 0, so the fine-tuning gain is
real recognition, not formatting.

KenLM weights on dev (wav2vec 2.0 + LM, sorted by dev WER):

| alpha | beta | dev WER | dev CER |
| --- | --- | --- | --- |
| 0.7 | 2.0 | 16.52% | 4.93% |
| 0.5 | 2.0 | 16.53% | 4.92% |
| 0.5 | 1.0 | 16.62% | 4.95% |
| 0.7 | 1.0 | 16.63% | 4.96% |
| 0.5 | 0.0 | 16.68% | 4.97% |
| 0.7 | 0.0 | 16.79% | 4.99% |
| 0.3 | 1.0 | 16.80% | 4.97% |
| 1.0 | 2.0 | 16.84% | 5.01% |
| 0.3 | 2.0 | 16.84% | 4.97% |
| 0.3 | 0.0 | 16.95% | 5.02% |
| 1.0 | 1.0 | 17.05% | 5.07% |
| 1.0 | 0.0 | 17.40% | 5.14% |

### 9.3 Error analysis

`scripts/error_analysis.py` (outputs in `results/error_analysis/`), on the 3,031 test clips after normalization:

| System | Outputs with a loop | Outputs with Latin letters | Substitutions | Deletions | Insertions |
| --- | --- | --- | --- | --- | --- |
| Whisper-small, zero-shot | 13 | 29 | 33.0% | 3.1% | 6.5% |
| Whisper-large-v3, zero-shot | 1 | 2 | 17.4% | 1.5% | 2.8% |
| Whisper-small fine-tuned, beam 5 | 0 | 0 | 15.7% | 1.8% | 2.2% |
| wav2vec2 fine-tuned, greedy | 0 | 0 | 18.6% | 2.7% | 2.0% |
| wav2vec2 fine-tuned + KenLM | 0 | 0 | 13.9% | 2.2% | 2.2% |

(Substitutions, deletions and insertions are per reference word; a loop is a 1 to 3 word phrase repeated 4 times.)

Most frequent one-letter mistakes of fine-tuned Whisper (words that differ from the reference by exactly one letter):

| Reference letter | Output letter | Words |
| --- | --- | --- |
| ے | ی | 99 |
| ی | ے | 90 |
| ئ | ی | 85 |
| ہ | ے | 43 |
| ہ | ا | 42 |
| ا | ہ | 35 |

- **Zero-shot Whisper breaks in ways fine-tuning removes:** 13 hallucination loops and 29 outputs in Latin script
  for Whisper-small, 0 and 0 after fine-tuning.
- **The remaining errors are mostly spelling of words heard correctly:** the most common one-letter mistakes are
  letters that sound the same or nearly the same: ے / ی (bari ye and choti ye), ئ / ی, and word-final ہ / ے / ا.
- **This is what the LM fixes:** for wav2vec 2.0 it cuts substitutions from 18.6% to 13.9% of reference words, while
  deletions and insertions barely change. The LM chooses the correct spelling of a word; it does not recover missing words.

### 9.4 Limitations

- Read speech from one corpus only: these numbers say little about conversational or noisy Urdu.
- The best LM `beta` (2.0) is the edge of our dev grid. We did not extend the grid afterwards, because the test
  run was already done and test must not be used for tuning.
- RTF was measured on three kinds of hardware (RTX 4060, Tesla T4, CPU); compare it only within the same hardware.
- We use a 25-hour subset of Kathbath, not all of it, and the dev set during training is the first 300 dev clips.

## 10. How to reproduce

Python 3.11, `pip install -r requirements.txt`. Seed 42 everywhere (subset selection, split, training, bootstrap).
LM decoding also needs `pyctcdecode==0.5.0` and `kenlm==0.2.0`, which build on Linux or Colab, not Windows; the
Colab notebooks install them. Kaldi and ESPnet are set up as described in their baseline folders.

```
python scripts/download_kathbath_urdu.py --target-hours 25 --seed 42   # data/raw/ (already committed)
python scripts/clean.py                                                # data/clean/
python scripts/split.py                                                # data/splits/
# our systems (GPU)
python models/whisper_finetune/train.py
python models/whisper_finetune/transcribe.py --model checkpoints/whisper_small_ft/best --manifest data/splits/test.csv --out results/predictions/whisper_small_ft_beam5.csv --system whisper_small_ft_beam5 --num-beams 5
python models/wav2vec2_finetune/train.py --epochs 15
python models/wav2vec2_finetune/transcribe.py --model checkpoints/w2v2_xlsr_ft/best --manifest data/splits/test.csv --out results/predictions/w2v2_xlsr_ft_greedy.csv --system w2v2_xlsr_ft_greedy
# KenLM: open models/wav2vec2_finetune/kenlm_colab.ipynb in Colab and run all
# baselines: baselines/gmm_hmm/README.md, baselines/ctc_attention_rnnt/README.md, section 7 for the published models
python src/evaluate.py --split data/splits/test.csv --pred results/predictions/<system>.csv
python scripts/make_tables.py
python scripts/error_analysis.py
pytest tests/
```

## 11. Repository layout

| Path | Content |
| --- | --- |
| `README.md` | This file |
| `LICENSE` | MIT license for our own code (the dataset has its own license, see section 3) |
| `requirements.txt` | Pinned package versions (Python 3.11) |
| `.gitignore` | Files git ignores (caches, virtual environments, checkpoints) |
| `contributions/` | One file per member: `<NAME>_<STUDENT_ID>_<GITHUB_USERNAME>.md` |
| `docs/qa.md` | Likely questions about the project, with short answers (presentation preparation) |
| `data/raw/` | Kathbath Urdu audio (16-bit FLAC, by speaker) and `metadata.csv`, as downloaded |
| `data/clean/` | `metadata.csv` after cleaning, `cleaning_log.csv`, `cleaning_summary.csv`, `manual_review.csv` (listening notes), `profile*.csv` (audio profiling), `label_check*.csv` (Whisper label check) |
| `data/splits/` | `train.csv`, `dev.csv`, `test.csv`, `split_stats.csv`, `test_overlap.csv` |
| `scripts/download_kathbath_urdu.py` | Selects the 25-hour subset and writes `data/raw/` |
| `scripts/clean.py` | Cleaning checks (section 4) |
| `scripts/profile_data.py` | Audio profiling that picks clips to listen to |
| `scripts/find_label_errors.py` | Whisper-based transcript check |
| `scripts/split.py` | Speaker-disjoint split (section 5) |
| `scripts/make_kaldi_dirs.py` | Kaldi data folders for the GMM-HMM baseline |
| `scripts/make_tables.py` | Builds `results/tables.md`, the source of section 9's tables |
| `scripts/error_analysis.py` | Builds `results/error_analysis/` (section 9.3) |
| `src/normalize.py`, `src/urdu_numbers.py` | The shared normalization; digits to Urdu words |
| `src/evaluate.py` | WER, CER and bootstrap CI for every system; appends to `results/results.csv` |
| `baselines/gmm_hmm/` | Kaldi MFCC + GMM-HMM baseline (own README) |
| `baselines/ctc_attention_rnnt/` | ESPnet CTC, hybrid CTC/attention and RNN-T baselines (own README) |
| `baselines/pretrained/anuragshas_lm_colab.ipynb` | Runs the anuragshas checkpoint with its own LM on Colab |
| `models/whisper_finetune/` | `train.py` (fine-tuning), `transcribe.py` (decoding and timing for any Whisper model) |
| `models/wav2vec2_finetune/` | `train.py`, `transcribe.py` (greedy, our LM or a checkpoint's own LM), `build_lm.sh`, `tune_lm.py`, `kenlm_colab.ipynb` |
| `lm/train_3gram.arpa` | The KenLM 3-gram built from train text |
| `results/predictions/` | One prediction file per system and split (`utt_id,hyp`, raw output) |
| `results/timing/` | One timing file per system: model, revision, beam, batch, hardware, RTF |
| `results/results.csv` | WER, CER and CI of every system on every split |
| `results/tables.md` | Generated result tables |
| `results/lm_tuning_dev.csv` | The dev grid for the KenLM weights |
| `results/error_analysis/` | Generated error analysis |
| `tests/` | Tests for cleaning, splitting, normalization, numbers, evaluation and the baselines' helpers |
