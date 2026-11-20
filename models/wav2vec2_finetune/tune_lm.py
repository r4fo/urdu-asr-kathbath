"""Choose the KenLM weights (alpha, beta) for wav2vec 2.0 decoding on the DEV split.

Runs the acoustic model once on the manifest, then decodes the same outputs
with every (alpha, beta) pair and scores each with src/normalize.py + jiwer.
Never run this on the test split: test is decoded once, with the chosen pair,
by transcribe.py.

Output: results/lm_tuning_dev.csv  (alpha,beta,wer,cer), sorted by WER

Usage (repo root; needs pyctcdecode and kenlm, e.g. on Colab):
  python models/wav2vec2_finetune/tune_lm.py --model <hub id or folder> \
      --manifest data/splits/dev.csv --lm lm/train_3gram.arpa
"""

import argparse
import importlib.util
import itertools
import sys
from pathlib import Path

import jiwer
import pandas as pd
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.normalize import normalize  # noqa: E402

_spec = importlib.util.spec_from_file_location("transcribe", Path(__file__).with_name("transcribe.py"))
transcribe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(transcribe)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--manifest", default="data/splits/dev.csv")
    p.add_argument("--lm", required=True)
    p.add_argument("--alphas", default="0.3,0.5,0.7,1.0")
    p.add_argument("--betas", default="0.0,1.0,2.0")
    p.add_argument("--beam-width", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", default="results/lm_tuning_dev.csv")
    args = p.parse_args()
    if "test" in Path(args.manifest).name:
        raise SystemExit("tune_lm.py is for the dev split only; test is decoded once with transcribe.py")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(args.manifest, encoding="utf-8")
    processor = Wav2Vec2Processor.from_pretrained(args.model, revision=args.revision)
    model = Wav2Vec2ForCTC.from_pretrained(args.model, revision=args.revision).to(device).eval()

    # 1) acoustic model once: per-utterance log-probabilities, cut to the real length
    log_probs = []
    rows = list(df.itertuples(index=False))
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i:i + args.batch_size]
        audios = [transcribe.load_audio(r.path) for r in batch]
        inputs = processor(audios, sampling_rate=16000, return_tensors="pt", padding=True)
        mask = inputs.get("attention_mask")
        with torch.inference_mode():
            logits = model(inputs.input_values.to(device),
                           attention_mask=mask.to(device) if mask is not None else None).logits.float()
        frames = (model._get_feat_extract_output_lengths(mask.sum(-1)).tolist() if mask is not None
                  else [logits.shape[1]] * len(batch))
        lp = torch.log_softmax(logits, dim=-1).cpu().numpy()
        log_probs.extend(lp[j][:int(n)] for j, n in enumerate(frames))
    refs = [normalize(t) for t in df["text"]]
    print(f"Acoustic outputs ready for {len(log_probs)} utterances")

    # 2) decode with every (alpha, beta) pair
    results = []
    alphas = [float(a) for a in args.alphas.split(",")]
    betas = [float(b) for b in args.betas.split(",")]
    for alpha, beta in itertools.product(alphas, betas):
        decoder = transcribe.make_lm_decoder(processor, args.lm, alpha, beta)
        hyps = [normalize(decoder.decode(lp, beam_width=args.beam_width)) for lp in log_probs]
        wer, cer = jiwer.wer(refs, hyps), jiwer.cer(refs, hyps)
        results.append({"alpha": alpha, "beta": beta, "wer": round(wer, 5), "cer": round(cer, 5)})
        print(f"alpha {alpha:.2f} beta {beta:.2f}: WER {wer:.2%} CER {cer:.2%}")

    out = pd.DataFrame(results).sort_values("wer")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    best = out.iloc[0]
    print(f"Best on dev: alpha {best.alpha}, beta {best.beta} (WER {best.wer:.2%}). Wrote {args.out}")


if __name__ == "__main__":
    main()
