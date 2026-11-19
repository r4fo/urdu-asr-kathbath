# CTC, hybrid CTC/attention and RNN-T baselines (Member 4)

Three small end-to-end models trained by us with one toolkit, **ESPnet 202412** (git tag `v.202412`),
on the shared train split, with character tokens. They exist to compare the three model families
on the same data; they are small and trained for a few hours each, so strong numbers are not the goal.

| System | Prediction file | Training loss | Decoding |
| --- | --- | --- | --- |
| CTC | `results/predictions/ctc.csv` | CTC only | greedy (beam 1) |
| Hybrid CTC/attention | `results/predictions/hybrid.csv` | 0.3 CTC + 0.7 attention | joint CTC/attention beam search, beam 10 |
| RNN-T | `results/predictions/rnnt.csv` | RNN-T + 0.3 auxiliary CTC | transducer beam search, beam 10 |

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
| `setup_espnet.sh` | Installs ESPnet 202412 and torch 2.4.1, and clones the ESPnet recipe scripts |
| `prepare_data.py` | Split CSVs → ESPnet data folders (`wav.scp`, `text`, `utt2spk`, `spk2utt`); stops if a speaker is in two splits |
| `run.sh` | Runs ESPnet `asr.sh` stages 2-12 for one system and writes the prediction CSV and timing notes |
| `to_predictions.py` | ESPnet output → `utt_id,hyp` CSV (raw output, not normalized) |
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

## Hardware and results

TODO after training: GPU, training time per system, decoding RTF (from `work/full/timing_<system>.txt`),
and dev/test WER and CER from `results/results.csv` (computed by `src/evaluate.py`, not by ESPnet).
