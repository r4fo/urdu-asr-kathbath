"""Transcribe a manifest with a Whisper model and measure the real-time factor.

Works for zero-shot Hugging Face checkpoints (e.g. openai/whisper-small) and
for our fine-tuned checkpoints (a local folder).

Writes:
  --out                      prediction CSV: utt_id,hyp  (raw model output, not normalized)
  results/timing/<system>.csv  one row: system,model,revision,num_beams,batch_size,
                               hardware,audio_seconds,decode_seconds,rtf
                               (one file per system, so branches never conflict;
                               Member 2 combines them)

RTF = model decoding time / audio duration. Decoding time is measured around
model.generate() with torch.cuda.synchronize(), after one warm-up batch, and
does not include reading audio files from disk.

Usage (repo root):
  python models/whisper_finetune/transcribe.py --model openai/whisper-small \
      --manifest data/clean/metadata.csv --out results/predictions/whisper_small_zeroshot_all.csv \
      --system whisper_small_zeroshot_all
"""

import argparse
import csv
import platform
import time
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Hub id or local checkpoint folder")
    p.add_argument("--revision", default=None, help="Hub commit hash to pin (recommended)")
    p.add_argument("--manifest", required=True, help="CSV with utt_id,path[,duration]")
    p.add_argument("--out", required=True)
    p.add_argument("--system", required=True, help="name used in results and timing")
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="only the first N rows (for testing)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    hardware = torch.cuda.get_device_name(0) if device == "cuda" else platform.processor()

    df = pd.read_csv(args.manifest, encoding="utf-8")
    if args.limit:
        df = df.head(args.limit)
    if "duration" in df.columns:  # similar lengths in a batch -> less padding work
        df = df.sort_values("duration", ascending=False)

    processor = WhisperProcessor.from_pretrained(args.model, revision=args.revision)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, torch_dtype=dtype).to(device).eval()
    revision = args.revision or getattr(model.config, "_commit_hash", None) or "local"
    gen_kwargs = dict(language="urdu", task="transcribe",
                      num_beams=args.num_beams, max_new_tokens=225)

    rows = list(df.itertuples(index=False))
    batches = [rows[i:i + args.batch_size] for i in range(0, len(rows), args.batch_size)]

    def run(batch):
        audios = [load_audio(r.path) for r in batch]
        feats = processor.feature_extractor(
            audios, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            ids = model.generate(feats, **gen_kwargs)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        texts = processor.batch_decode(ids, skip_special_tokens=True)
        audio_sec = sum(len(a) for a in audios) / 16000
        return texts, elapsed, audio_sec

    run(batches[0][:2])  # warm-up, not timed

    hyps, decode_sec, audio_sec = {}, 0.0, 0.0
    for i, batch in enumerate(batches, 1):
        texts, elapsed, seconds = run(batch)
        decode_sec += elapsed
        audio_sec += seconds
        for r, t in zip(batch, texts):
            hyps[r.utt_id] = t.strip()
        if i % 20 == 0 or i == len(batches):
            print(f"  {i}/{len(batches)} batches, RTF so far {decode_sec / audio_sec:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sorted(hyps.items()), columns=["utt_id", "hyp"]).to_csv(
        out, index=False, encoding="utf-8")

    TIMING_DIR.mkdir(parents=True, exist_ok=True)
    with (TIMING_DIR / f"{args.system}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=TIMING_FIELDS)
        w.writeheader()
        w.writerow({"system": args.system, "model": args.model, "revision": revision,
                    "num_beams": args.num_beams, "batch_size": args.batch_size,
                    "hardware": hardware, "audio_seconds": round(audio_sec, 2),
                    "decode_seconds": round(decode_sec, 2),
                    "rtf": round(decode_sec / audio_sec, 5)})
    print(f"Wrote {out} ({len(hyps)} utterances). RTF {decode_sec / audio_sec:.4f} on {hardware}, "
          f"revision {revision}")


if __name__ == "__main__":
    main()
