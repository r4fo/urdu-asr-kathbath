"""Transcribe a manifest with a wav2vec 2.0 CTC model, greedy or with an n-gram LM.

Works for our fine-tuned checkpoint (a local folder) and for published Urdu CTC
checkpoints from the Hub.

Writes:
  --out                         prediction CSV: utt_id,hyp (raw output, not normalized)
  results/timing/<system>.csv   system,model,revision,num_beams,batch_size,hardware,
                                audio_seconds,decode_seconds,rtf

RTF covers the model forward pass and the decoding (greedy or LM beam search),
measured with torch.cuda.synchronize() after one warm-up batch; reading audio
from disk is not counted.

LM decoding needs pyctcdecode and kenlm (pip install pyctcdecode kenlm; easiest on
Linux, WSL or Colab). The LM must be trained on train-split text only. Choose
--alpha and --beta on dev, never on test.

Usage (repo root):
  python models/wav2vec2_finetune/transcribe.py --model checkpoints/w2v2_xlsr_ft/best \
      --manifest data/splits/dev.csv --out results/predictions/w2v2_ft_dev.csv --system w2v2_ft_dev
  # with the LM:
  ... --lm lm/train_3gram.arpa --alpha 0.5 --beta 1.0 --beam-width 50
"""

import argparse
import csv
import platform
import time
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

TIMING_DIR = Path("results/timing")
TIMING_FIELDS = ["system", "model", "revision", "num_beams", "batch_size",
                 "hardware", "audio_seconds", "decode_seconds", "rtf"]


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"{path}: expected 16 kHz, got {sr}")
    return audio


def make_lm_decoder(processor, lm_path, alpha, beta):
    from pyctcdecode import build_ctcdecoder
    vocab = processor.tokenizer.get_vocab()
    labels = [""] * len(vocab)
    for token, idx in vocab.items():
        labels[idx] = token
    pad, delim = processor.tokenizer.pad_token, processor.tokenizer.word_delimiter_token
    labels = ["" if t == pad else " " if t == delim else t for t in labels]
    return build_ctcdecoder(labels, kenlm_model_path=lm_path, alpha=alpha, beta=beta)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--system", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--lm", default=None, help="ARPA or binary KenLM file (train-split text only)")
    p.add_argument("--alpha", type=float, default=0.5, help="LM weight (tune on dev)")
    p.add_argument("--beta", type=float, default=1.0, help="word insertion bonus (tune on dev)")
    p.add_argument("--beam-width", type=int, default=50)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hardware = torch.cuda.get_device_name(0) if device == "cuda" else platform.processor()
    df = pd.read_csv(args.manifest, encoding="utf-8")
    if args.limit:
        df = df.head(args.limit)
    if "duration" in df.columns:
        df = df.sort_values("duration", ascending=False)

    processor = Wav2Vec2Processor.from_pretrained(args.model, revision=args.revision)
    model = Wav2Vec2ForCTC.from_pretrained(args.model, revision=args.revision).to(device).eval()
    if device == "cuda":
        model = model.half()
    revision = args.revision or getattr(model.config, "_commit_hash", None) or "local"
    decoder = make_lm_decoder(processor, args.lm, args.alpha, args.beta) if args.lm else None

    rows = list(df.itertuples(index=False))
    batches = [rows[i:i + args.batch_size] for i in range(0, len(rows), args.batch_size)]

    def run(batch):
        audios = [load_audio(r.path) for r in batch]
        inputs = processor(audios, sampling_rate=16000, return_tensors="pt", padding=True)
        values = inputs.input_values.to(device, model.dtype)
        mask = inputs.get("attention_mask")
        mask = mask.to(device) if mask is not None else None
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            logits = model(values, attention_mask=mask).logits.float()
        if decoder is None:
            texts = processor.batch_decode(logits.argmax(dim=-1).cpu().numpy())
        else:
            frames = model._get_feat_extract_output_lengths(mask.sum(-1)) if mask is not None \
                else [logits.shape[1]] * len(audios)
            log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
            texts = [decoder.decode(lp[:int(n)], beam_width=args.beam_width)
                     for lp, n in zip(log_probs, frames)]
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return texts, elapsed, sum(len(a) for a in audios) / 16000

    run(batches[0][:2])  # warm-up, not timed

    hyps, decode_sec, audio_sec = {}, 0.0, 0.0
    for i, batch in enumerate(batches, 1):
        texts, elapsed, seconds = run(batch)
        decode_sec += elapsed
        audio_sec += seconds
        for r, t in zip(batch, texts):
            hyps[r.utt_id] = " ".join(t.split())
        if i % 20 == 0 or i == len(batches):
            print(f"  {i}/{len(batches)} batches, RTF so far {decode_sec / audio_sec:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sorted(hyps.items()), columns=["utt_id", "hyp"]).to_csv(out, index=False, encoding="utf-8")
    TIMING_DIR.mkdir(parents=True, exist_ok=True)
    with (TIMING_DIR / f"{args.system}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=TIMING_FIELDS)
        w.writeheader()
        w.writerow({"system": args.system, "model": args.model, "revision": revision,
                    "num_beams": args.beam_width if decoder else 1, "batch_size": args.batch_size,
                    "hardware": hardware, "audio_seconds": round(audio_sec, 2),
                    "decode_seconds": round(decode_sec, 2), "rtf": round(decode_sec / audio_sec, 5)})
    print(f"Wrote {out} ({len(hyps)} utterances). RTF {decode_sec / audio_sec:.4f} on {hardware}, "
          f"{'LM ' + args.lm if decoder else 'greedy'}, revision {revision}")


if __name__ == "__main__":
    main()
