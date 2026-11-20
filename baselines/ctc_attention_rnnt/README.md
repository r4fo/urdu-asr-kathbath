# CTC, hybrid CTC/attention and RNN-T baselines (Member 4)

Three small end-to-end models trained by us with one toolkit, **ESPnet 202511** (git tag `v.202511`),
on the shared train split, with character tokens. They exist to compare the three model families
on the same data; they are small and trained for a few hours each, so strong numbers are not the goal.

| System | Test predictions | Dev predictions | Training loss | Decoding | Epochs |
| --- | --- | --- | --- | --- | --- |
| CTC | `results/predictions/ctc.csv` | `results/predictions/ctc_dev.csv` | CTC only | greedy (beam 1) | 20 |
| Hybrid CTC/attention | `results/predictions/hybrid.csv` | `results/predictions/hybrid_dev.csv` | 0.3 CTC + 0.7 attention | joint CTC/attention beam search, beam 10 | 20 |
| RNN-T | `results/predictions/rnnt.csv` | `results/predictions/rnnt_dev.csv` | RNN-T + 0.3 auxiliary CTC | transducer beam search, beam 10 | 20 (resumed once, see below) |

No language model is used by any of the three.

## Shared settings

All three use the same encoder, features and training schedule, so only the decoder and loss differ:

- Features: 80-dim log-mel (25 ms window, 10 ms hop), global mean/variance normalization, SpecAugment
- Encoder: Conformer, 8 blocks, dimension 144, 4 attention heads, conv kernel 15 (4.6M parameters for the CTC model)
- Hybrid decoder: Transformer, 4 blocks
- RNN-T: 1-layer LSTM prediction network (256), joint network (256)
- Tokens: characters of the training transcripts (`token_type char`), 43 symbols on the 50-clip sample
- Adam, peak learning rate 0.002, 2000 warm-up steps, 20 epochs, mixed precision, seed 42
- Best epoch: the one with the lowest dev loss (`valid.loss.best.pth`). Test is decoded once, after that.
- Clips longer than 30 s are left out of training

Configs: `conf/train_{ctc,hybrid,rnnt}.yaml` and `conf/decode_{ctc,hybrid,rnnt}.yaml`.

The RNN-T loss is computed with `torchaudio.functional.rnnt_loss` through the small drop-in module
`rnnt_loss/warprnnt_pytorch/`, because ESPnet's default (warp-transducer) must be compiled from source.
It is the same standard RNN-T loss. `run.sh` uses it only if warp-transducer is not installed.

## Files

| File | What it does |
| --- | --- |
| `setup_espnet.sh` | Clones the ESPnet 202511 recipe scripts and installs ESPnet and torch 2.4.1 in its own Python 3.12 environment (made with uv), because Colab's Python 3.13 has no wheels for them |
| `prepare_data.py` | Split CSVs → ESPnet data folders (`wav.scp`, `text`, `utt2spk`, `spk2utt`); stops if a speaker is in two splits |
| `run.sh` | Runs ESPnet `asr.sh` stages 2-12 for one system and writes the prediction CSV and timing notes |
| `to_predictions.py` | ESPnet output → `utt_id,hyp` CSV (raw output, not normalized) |
| `make_timing.py` | ESPnet's test RTF log → `results/timing/<system>.csv` (team timing format) |
| `make_sample_splits.py` | 40/10/10-clip speaker-disjoint sample from `data/clean/` for testing the pipeline |
| `colab.ipynb` | The same steps on Google Colab, with checkpoints on Google Drive |
| `conf/` | Training and decoding configs |
| `rnnt_loss/` | RNN-T loss drop-in (see above) |

`work/` (data folders, dumped audio, checkpoints) is created by `run.sh` and ignored by git.
Tests: `tests/test_ctc_attention_rnnt.py`.

## How to run

```
bash baselines/ctc_attention_rnnt/setup_espnet.sh            # once (Colab: every session)
python3 baselines/ctc_attention_rnnt/make_sample_splits.py   # pipeline test, 2 epochs
bash baselines/ctc_attention_rnnt/run.sh ctc --name sample \
    --splits-dir baselines/ctc_attention_rnnt/work/sample_splits --asr-args "--max_epoch 2"

bash baselines/ctc_attention_rnnt/run.sh ctc                 # real run on data/splits/, dev only
bash baselines/ctc_attention_rnnt/run.sh ctc --stage 12 --test   # once, after settings are frozen
```

Repeat with `hybrid` and `rnnt`. Training resumes from the last checkpoint when `run.sh` is run again with `--stage 11`.

## RNN-T training was interrupted once

RNN-T training stopped at epoch 10 when Colab's GPU usage limit was reached. It was resumed from the
last checkpoint on Google Drive with ESPnet `asr.sh` stage 11 and trained to the same 20 epochs as CTC and
hybrid:

```
bash baselines/ctc_attention_rnnt/run.sh rnnt --expdir <drive>/exp --stage 11 --asr-args "--max_epoch 20"
```

ESPnet's resume restores the model, the optimizer and the learning-rate scheduler, so the run follows the
same 20-epoch schedule as an uninterrupted one. Only the predictions of the final 20-epoch model are
reported. Test was also decoded once with the 10-epoch model before the retrain; those predictions were
overwritten and never scored, so no setting was chosen by looking at test.

## Hardware and results

All training and decoding ran on Google Colab with one NVIDIA Tesla T4 (15 GB) and an Intel Xeon CPU @ 2.00GHz.

| System | Training time | Dev decoding | Dev RTF | Test decoding | Test RTF |
| --- | --- | --- | --- | --- | --- |
| CTC (greedy) | 2696 s | 2343 s | 0.240 | 3997 s | 0.240 |
| Hybrid (beam 10) | 3183 s | 4147 s | 0.424 | 6719 s | 0.404 |
| RNN-T (beam 10) | 1220 s + 1663 s for the two resumed parts (see above) | 3080 s | 0.315 | 5074 s | 0.305 |

Decoding time and RTF come from ESPnet's own per-utterance timing (`calculate_rtf.log`: sum of the
decoding time of each clip, one job, batch size 1, model loading not counted), over 1956 dev clips and
3031 test clips (16636 s of test audio). The test rows are also in `results/timing/<system>.csv`, written
by `make_timing.py`. Training time is ESPnet's elapsed time for stage 11 from `train.log`
(collected by `run.sh` into `work/full/timing_<system>.txt`). For RNN-T, `train.log` is rewritten on each
resume, so only the last two training parts were logged; the time before the first interruption was not
recorded.

Dev and test WER and CER: see `results/results.csv` (computed by `src/evaluate.py`, not by ESPnet).
