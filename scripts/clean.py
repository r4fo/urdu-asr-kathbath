"""Clean the Kathbath Urdu subset: data/raw/ -> data/clean/.

Finds and fixes real problems; every removed or changed utterance is logged
with its reason in data/clean/cleaning_log.csv.

Audio is never modified (only filtered), so data/clean/metadata.csv points to
the audio files in data/raw/audio/ instead of storing a second copy.

Outputs:
  data/clean/metadata.csv          kept utterances: utt_id,path,speaker_id,gender,duration,text,source_split
  data/clean/cleaning_log.csv      utt_id,action,reason,detail   (action = removed | fixed | flagged)
  data/clean/cleaning_summary.csv  reason,action,utterances,hours

Usage (repo root):
  python scripts/clean.py
"""

import argparse
import hashlib
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

RAW_META = Path("data/raw/metadata.csv")
CLEAN_DIR = Path("data/clean")

# Arabic-script letter variants that should be written with the Urdu form.
LETTER_FIXES = {
    "ي": "ی",  # ARABIC YEH -> FARSI YEH (Urdu choti ye)
    "ى": "ی",  # ALEF MAKSURA -> FARSI YEH
    "ك": "ک",  # ARABIC KAF -> KEHEH (Urdu kaf)
    "ه": "ہ",  # ARABIC HEH -> HEH GOAL (Urdu choti he)
}
# Arabic-Indic digits -> Extended Arabic-Indic (Urdu) digits.
DIGIT_FIXES = {chr(0x0660 + i): chr(0x06F0 + i) for i in range(10)}
# Invisible characters that carry no meaning in transcripts.
INVISIBLE = ["​", "‎", "‏", "﻿", "ـ"]  # ZWSP, LRM, RLM, BOM, tatweel

LATIN = re.compile(r"[A-Za-z]")
URDU_LETTER = re.compile(r"[؀-ۿݐ-ݿ]")


def fix_text(text):
    """Return (fixed_text, list_of_fix_names)."""
    fixes = []
    original = text
    text = unicodedata.normalize("NFC", text)
    if text != original:
        fixes.append("unicode_nfc")
    for bad, good in {**LETTER_FIXES, **DIGIT_FIXES}.items():
        if bad in text:
            text = text.replace(bad, good)
            fixes.append(f"U+{ord(bad):04X}->U+{ord(good):04X}")
    for ch in INVISIBLE:
        if ch in text:
            text = text.replace(ch, "")
            fixes.append(f"removed_U+{ord(ch):04X}")
    collapsed = re.sub(r"\s+", " ", text).strip()
    if collapsed != text:
        fixes.append("whitespace")
    return collapsed, fixes


def chars_per_second(text, seconds):
    letters = len(URDU_LETTER.findall(text))
    return letters / seconds if seconds > 0 else float("inf")


def audio_checks(audio):
    """Return (rms_dbfs, clipped_fraction) for a mono float array."""
    if audio.size == 0:
        return float("-inf"), 0.0
    rms = float(np.sqrt(np.mean(np.square(audio))))
    rms_db = 20 * np.log10(rms) if rms > 0 else float("-inf")
    clipped = float(np.mean(np.abs(audio) >= 0.999))
    return rms_db, clipped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-sec", type=float, default=0.5)
    p.add_argument("--max-sec", type=float, default=30.0)
    p.add_argument("--silence-db", type=float, default=-50.0)
    p.add_argument("--max-clipped", type=float, default=0.01)
    p.add_argument("--min-cps", type=float, default=3.0)
    p.add_argument("--max-cps", type=float, default=30.0)
    args = p.parse_args()

    meta = pd.read_csv(RAW_META, encoding="utf-8")
    log, keep_rows, seen_hashes = [], [], {}

    def record(utt, action, reason, detail=""):
        log.append({"utt_id": utt, "action": action, "reason": reason, "detail": detail})

    for i, row in enumerate(meta.itertuples(index=False), 1):
        utt, path = row.utt_id, Path(row.path)
        try:
            data = path.read_bytes()
            audio, sr = sf.read(path, dtype="float32", always_2d=False)
        except Exception as exc:  # missing or unreadable file
            record(utt, "removed", "unreadable_audio", str(exc)[:200])
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        digest = hashlib.md5(data).hexdigest()
        if digest in seen_hashes:
            record(utt, "removed", "duplicate_audio", f"same file as {seen_hashes[digest]}")
            continue
        seen_hashes[digest] = utt

        seconds = len(audio) / sr
        if abs(seconds - row.duration) > 0.1:
            record(utt, "fixed", "duration_mismatch", f"metadata {row.duration:.2f}s, file {seconds:.2f}s")
        if sr != 16000:
            record(utt, "removed", "wrong_sample_rate", str(sr))
            continue
        if seconds < args.min_sec or seconds > args.max_sec:
            record(utt, "removed", "duration_out_of_range", f"{seconds:.2f}s")
            continue

        rms_db, clipped = audio_checks(audio)
        if rms_db < args.silence_db:
            record(utt, "removed", "silent_audio", f"rms {rms_db:.1f} dBFS")
            continue
        if clipped > args.max_clipped:
            record(utt, "removed", "clipped_audio", f"{clipped:.2%} samples clipped")
            continue

        text = row.text if isinstance(row.text, str) else ""
        text, fixes = fix_text(text)
        if not text:
            record(utt, "removed", "empty_transcript")
            continue
        if LATIN.search(text):
            record(utt, "removed", "latin_letters_in_transcript", text[:80])
            continue
        cps = chars_per_second(text, seconds)
        if cps < args.min_cps or cps > args.max_cps:
            record(utt, "removed", "text_audio_mismatch", f"{cps:.1f} letters/s")
            continue
        for fix in fixes:
            record(utt, "fixed", "transcript_characters", fix)

        keep_rows.append({"utt_id": utt, "path": row.path, "speaker_id": row.speaker_id,
                          "gender": row.gender, "duration": round(seconds, 4), "text": text,
                          "source_split": row.source_split})
        if i % 2000 == 0:
            print(f"  checked {i}/{len(meta)}")

    clean = pd.DataFrame(keep_rows)
    dup_text = clean[clean["text"].duplicated(keep=False)]
    for utt in dup_text["utt_id"]:
        record(utt, "flagged", "duplicate_text", "same sentence as another utterance (kept)")

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_DIR / "metadata.csv", index=False, encoding="utf-8")
    log_df = pd.DataFrame(log, columns=["utt_id", "action", "reason", "detail"])
    log_df.to_csv(CLEAN_DIR / "cleaning_log.csv", index=False, encoding="utf-8")

    hours = meta.set_index("utt_id")["duration"] / 3600
    summary = (log_df.drop_duplicates(["utt_id", "action", "reason"])
               .assign(hours=lambda d: d["utt_id"].map(hours))
               .groupby(["reason", "action"])
               .agg(utterances=("utt_id", "count"), hours=("hours", "sum"))
               .reset_index().round(3))
    summary.to_csv(CLEAN_DIR / "cleaning_summary.csv", index=False, encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"\nRaw:   {len(meta)} utterances, {meta['duration'].sum() / 3600:.2f} h")
    print(f"Clean: {len(clean)} utterances, {clean['duration'].sum() / 3600:.2f} h, "
          f"{clean['speaker_id'].nunique()} speakers")


if __name__ == "__main__":
    main()
