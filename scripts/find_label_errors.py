"""Find utterances whose transcript may not match the audio.

Compares a zero-shot Whisper prediction with the reference for every utterance
(both normalized with src/normalize.py, as in scoring) and ranks the mismatches.
Whisper output is NOT treated as ground truth: it only ranks candidates, and a
person listens before anything is removed (decisions go to
data/clean/manual_review.csv, which scripts/clean.py applies).

Zero-shot Whisper sometimes gets stuck repeating a word or phrase ("hallucination
loop"). Those clips have a very high CER because of the model, not the reference,
so they are marked and left out of the candidate list.

Outputs:
  data/clean/label_check.csv             every utterance: split, cer, len_ratio, loop
  data/clean/label_check_candidates.csv  the highest-CER utterances without a loop

Usage (repo root):
  python scripts/find_label_errors.py --pred results/predictions/whisper_small_zeroshot_all.csv
"""

import argparse
import sys
from pathlib import Path

import jiwer
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.normalize import normalize  # noqa: E402


def has_loop(text, min_repeats=4):
    """True if the same 1-3 word phrase appears at least min_repeats times in a row."""
    words = text.split()
    for n in (1, 2, 3):
        for i in range(len(words) - min_repeats * n + 1):
            phrase = words[i:i + n]
            if all(words[i + k * n:i + (k + 1) * n] == phrase for k in range(1, min_repeats)):
                return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--meta", default="data/clean/metadata.csv")
    p.add_argument("--splits", default="data/splits")
    p.add_argument("--out", default="data/clean/label_check.csv")
    p.add_argument("--candidates", default="data/clean/label_check_candidates.csv")
    p.add_argument("--top", type=int, default=30)
    args = p.parse_args()

    meta = pd.read_csv(args.meta, encoding="utf-8")
    pred = pd.read_csv(args.pred, encoding="utf-8").fillna({"hyp": ""})
    df = meta.merge(pred, on="utt_id", how="inner")
    split_of = {}
    for name in ("train", "dev", "test"):
        for utt in pd.read_csv(Path(args.splits) / f"{name}.csv", encoding="utf-8")["utt_id"]:
            split_of[utt] = name
    df["split"] = df["utt_id"].map(split_of)

    df["ref_n"] = df["text"].map(normalize)
    df["hyp_n"] = df["hyp"].map(normalize)
    df["cer"] = [jiwer.cer(r, h) for r, h in zip(df["ref_n"], df["hyp_n"])]
    df["len_ratio"] = df["hyp_n"].str.len() / df["ref_n"].str.len().clip(lower=1)
    df["loop"] = df["hyp_n"].map(has_loop)

    print(f"Compared {len(df)} utterances (of {len(meta)} in the metadata)")
    print(f"Hallucination loops: {int(df['loop'].sum())} ({df['loop'].mean():.2%}); by split: "
          f"{df.loc[df['loop'], 'split'].value_counts().to_dict()}")

    # Corpus numbers only on train + dev: the test split is scored once, at the end.
    seen = df[df["split"] != "test"]
    for label, part in (("train+dev", seen), ("train+dev without loops", seen[~seen["loop"]])):
        print(f"{label}: WER {jiwer.wer(list(part['ref_n']), list(part['hyp_n'])):.3f}, "
              f"CER {jiwer.cer(list(part['ref_n']), list(part['hyp_n'])):.3f}")

    cols = ["utt_id", "split", "speaker_id", "duration", "cer", "len_ratio", "loop"]
    df[cols].round(4).to_csv(args.out, index=False, encoding="utf-8")
    cand = df[~df["loop"]].nlargest(args.top, "cer")[cols[:-1] + ["path", "text", "hyp"]]
    cand.to_csv(args.candidates, index=False, encoding="utf-8")
    print(f"Wrote {args.out} and the {args.top} highest-CER clips without a loop to {args.candidates}")


if __name__ == "__main__":
    main()
