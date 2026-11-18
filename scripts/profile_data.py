"""Profile the raw Kathbath Urdu subset to find real data problems.

Prints distributions and the most extreme utterances, and writes
data/clean/profile.csv (one row per utterance with all measured values)
and data/clean/profile_candidates.csv (the most suspicious utterances, to
listen to before deciding cleaning rules).

Usage (repo root):
  python scripts/profile_data.py
"""

import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

RAW_META = Path("data/raw/metadata.csv")
OUT_DIR = Path("data/clean")
URDU_LETTER = re.compile(r"[؀-ۿݐ-ݿ]")
QUANTILES = [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]


def speech_profile(audio, sr, frame_ms=25, hop_ms=10):
    """Energy-based speech activity: leading/trailing silence (s) and speech ratio."""
    frame, hop = int(sr * frame_ms / 1000), int(sr * hop_ms / 1000)
    if len(audio) < frame:
        return len(audio) / sr, len(audio) / sr, 0.0
    n = 1 + (len(audio) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    energy = np.sqrt(np.mean(np.square(audio[idx]), axis=1)) + 1e-10
    db = 20 * np.log10(energy)
    threshold = max(-45.0, db.max() - 35.0)
    speech = db > threshold
    if not speech.any():
        return len(audio) / sr, len(audio) / sr, 0.0
    first, last = np.argmax(speech), n - 1 - np.argmax(speech[::-1])
    lead = first * hop / sr
    trail = max(0.0, (len(audio) - (last * hop + frame)) / sr)
    return lead, trail, float(speech.mean())


def filename_parts(utt_id):
    """Kathbath file names look like <id>-<speaker>-<m|f>."""
    parts = str(utt_id).split("-")
    if len(parts) >= 3 and parts[-1].lower() in {"m", "f"}:
        return parts[-2], parts[-1].lower()
    return None, None


def main():
    meta = pd.read_csv(RAW_META, encoding="utf-8")
    rows = []
    for i, r in enumerate(meta.itertuples(index=False), 1):
        audio, sr = sf.read(r.path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        seconds = len(audio) / sr
        lead, trail, ratio = speech_profile(audio, sr)
        text = r.text if isinstance(r.text, str) else ""
        letters = len(URDU_LETTER.findall(text))
        speech_sec = max(seconds - lead - trail, 1e-3)
        f_spk, f_gender = filename_parts(r.utt_id)
        rows.append({
            "utt_id": r.utt_id, "speaker_id": r.speaker_id, "gender": r.gender,
            "seconds": seconds, "words": len(text.split()), "letters": letters,
            "cps": letters / seconds, "cps_speech": letters / speech_sec,
            "lead_sil": lead, "trail_sil": trail, "speech_ratio": ratio,
            "peak": float(np.abs(audio).max()) if audio.size else 0.0,
            "fname_speaker_ok": f_spk is None or str(f_spk) == str(r.speaker_id),
            "fname_gender_ok": f_gender is None or f_gender == str(r.gender)[:1].lower(),
            "text": text,
        })
        if i % 2000 == 0:
            print(f"  profiled {i}/{len(meta)}")
    prof = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prof.to_csv(OUT_DIR / "profile.csv", index=False, encoding="utf-8")

    print("\n=== Distributions (quantiles) ===")
    cols = ["seconds", "words", "cps", "cps_speech", "lead_sil", "trail_sil", "speech_ratio", "peak"]
    print(prof[cols].quantile(QUANTILES).round(3).to_string())

    print("\n=== File name vs metadata ===")
    print("speaker id mismatches:", int((~prof["fname_speaker_ok"]).sum()))
    print("gender mismatches:    ", int((~prof["fname_gender_ok"]).sum()))

    print("\n=== Characters used in transcripts (rarest first) ===")
    chars = Counter("".join(prof["text"]))
    for ch, n in sorted(chars.items(), key=lambda kv: kv[1])[:40]:
        name = unicodedata.name(ch, "UNKNOWN")
        print(f"  U+{ord(ch):04X} {name:<40} {n}")
    print(f"  ({len(chars)} distinct characters in total)")

    # Robust outlier score on speaking rate: distance from the median in MADs.
    med = prof["cps_speech"].median()
    mad = (prof["cps_speech"] - med).abs().median() or 1e-6
    prof["cps_mad"] = (prof["cps_speech"] - med) / mad
    cand = pd.concat([
        prof.nsmallest(15, "cps_mad").assign(why="slowest speech rate"),
        prof.nlargest(15, "cps_mad").assign(why="fastest speech rate"),
        prof.nlargest(10, "lead_sil").assign(why="longest leading silence"),
        prof.nlargest(10, "trail_sil").assign(why="longest trailing silence"),
        prof.nsmallest(10, "speech_ratio").assign(why="lowest speech ratio"),
        prof[~prof["fname_speaker_ok"] | ~prof["fname_gender_ok"]].assign(why="file name mismatch"),
    ])
    cand.to_csv(OUT_DIR / "profile_candidates.csv", index=False, encoding="utf-8")
    print(f"\nMedian speech rate {med:.1f} letters/s, MAD {mad:.2f}")
    print(f"Wrote {OUT_DIR / 'profile.csv'} and {OUT_DIR / 'profile_candidates.csv'} "
          f"({len(cand)} candidates to listen to)")


if __name__ == "__main__":
    main()
