"""Find utterances whose transcript may not match the audio.

Compares a zero-shot Whisper prediction with the reference transcript for every
utterance and lists the ones with the highest character error rate (CER).
Whisper output is NOT treated as ground truth: it only ranks candidates, and a
person listens to each candidate before anything is removed (decisions go to
data/clean/manual_review.csv, which scripts/clean.py applies).

The light normalization here only makes the comparison fair (same letter forms,
no punctuation). Official scoring uses src/normalize.py.

Usage (repo root):
  python scripts/find_label_errors.py --pred results/predictions/whisper_small_zeroshot_all.csv
"""

import argparse
import re
import unicodedata

import jiwer
import pandas as pd

LETTER_MAP = str.maketrans({
    "آ": "ا", "أ": "ا", "إ": "ا",  # alef forms -> alef
    "ؤ": "و",  # waw with hamza -> waw
    "ۂ": "ہ", "ۃ": "ہ",  # heh goal forms -> heh goal
    "ۓ": "ے",  # yeh barree with hamza -> yeh barree
    "ي": "ی", "ى": "ی",  # Arabic yeh, alef maksura -> Farsi yeh
    "ك": "ک",  # Arabic kaf -> keheh
    "ه": "ہ",  # Arabic heh -> heh goal
})
DIACRITICS = re.compile(r"[ً-ْٰٓ-ٕ]")
NOT_LETTER = re.compile(r"[^؀-ۿݐ-ݿ\s]")


def light_normalize(text):
    text = unicodedata.normalize("NFC", str(text)).translate(LETTER_MAP)
    text = DIACRITICS.sub("", text)
    text = NOT_LETTER.sub(" ", text)
    text = re.sub(r"[،؛؟۔]", " ", text)  # Urdu punctuation
    return re.sub(r"\s+", " ", text).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--meta", default="data/clean/metadata.csv")
    p.add_argument("--out", default="data/clean/label_check_candidates.csv")
    p.add_argument("--top", type=int, default=30)
    args = p.parse_args()

    meta = pd.read_csv(args.meta, encoding="utf-8")
    pred = pd.read_csv(args.pred, encoding="utf-8").fillna({"hyp": ""})
    df = meta.merge(pred, on="utt_id", how="inner")
    df["ref_n"] = df["text"].map(light_normalize)
    df["hyp_n"] = df["hyp"].map(light_normalize)
    df["cer"] = [jiwer.cer(r, h) if r else 1.0 for r, h in zip(df["ref_n"], df["hyp_n"])]
    df["len_ratio"] = df["hyp_n"].str.len() / df["ref_n"].str.len().clip(lower=1)

    print(f"Compared {len(df)} utterances (of {len(meta)} in the metadata)")
    print("Per-utterance CER quantiles:")
    print(df["cer"].quantile([0.1, 0.5, 0.9, 0.99, 1.0]).round(3).to_string())
    print(f"Corpus CER (light normalization, all data, informational only): "
          f"{jiwer.cer(list(df['ref_n']), list(df['hyp_n'])):.3f}")

    cand = df.nlargest(args.top, "cer")[
        ["utt_id", "speaker_id", "duration", "cer", "len_ratio", "path", "text", "hyp"]]
    cand.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Wrote the {args.top} highest-CER utterances to {args.out}")


if __name__ == "__main__":
    main()
