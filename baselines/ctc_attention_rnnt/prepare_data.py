"""Turn the shared split CSVs into ESPnet (Kaldi-style) data folders.

Input:  <splits-dir>/train.csv, dev.csv and (optional) test.csv with columns
        utt_id,path,speaker_id,gender,duration,text
Output: <out-dir>/<split>/wav.scp, text, utt2spk, spk2utt

Kaldi tools need utterance IDs that start with the speaker ID, and sorting by
utterance must also sort by speaker. So every ID becomes "S<5-digit speaker>_<utt_id>".
to_predictions.py removes this prefix again.

The transcript is written as it is in the split file (not normalized), as the
team's format rules say. Final scores come only from src/evaluate.py.

Usage:
  python baselines/ctc_attention_rnnt/prepare_data.py \
      --splits-dir data/splits --out-dir baselines/ctc_attention_rnnt/work/data
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPLITS = ("train", "dev", "test")
COLUMNS = ["utt_id", "path", "speaker_id", "gender", "duration", "text"]


def kaldi_id(speaker_id, utt_id):
    return f"S{int(speaker_id):05d}_{utt_id}"


def read_split(csv_path):
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"{csv_path}: missing columns {missing}")
        return list(reader)


def write_split(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    for row in rows:
        text = " ".join(row["text"].split())  # one line, single spaces
        if not text:
            print(f"skip {row['utt_id']}: empty text", file=sys.stderr)
            continue
        audio = os.path.join(REPO_ROOT, row["path"])
        if not os.path.isfile(audio):
            sys.exit(f"audio not found: {audio}")
        entries.append((kaldi_id(row["speaker_id"], row["utt_id"]), audio,
                        text, f"S{int(row['speaker_id']):05d}"))
    entries.sort()  # byte order, same as LC_ALL=C sort

    ids = [e[0] for e in entries]
    if len(ids) != len(set(ids)):
        sys.exit(f"{out_dir}: duplicate utt_id")

    spk2utt = defaultdict(list)
    with open(os.path.join(out_dir, "wav.scp"), "w", encoding="utf-8") as wav, \
         open(os.path.join(out_dir, "text"), "w", encoding="utf-8") as txt, \
         open(os.path.join(out_dir, "utt2spk"), "w", encoding="utf-8") as u2s:
        for uid, audio, text, spk in entries:
            wav.write(f"{uid} {audio}\n")
            txt.write(f"{uid} {text}\n")
            u2s.write(f"{uid} {spk}\n")
            spk2utt[spk].append(uid)
    with open(os.path.join(out_dir, "spk2utt"), "w", encoding="utf-8") as s2u:
        for spk in sorted(spk2utt):
            s2u.write(f"{spk} {' '.join(spk2utt[spk])}\n")
    return len(entries), len(spk2utt)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--splits-dir", default=os.path.join(REPO_ROOT, "data", "splits"))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    speakers = {}
    for split in SPLITS:
        csv_path = os.path.join(args.splits_dir, f"{split}.csv")
        if not os.path.isfile(csv_path):
            if split == "test":
                print("no test.csv, skipping test")
                continue
            sys.exit(f"missing {csv_path}")
        rows = read_split(csv_path)
        speakers[split] = {r["speaker_id"] for r in rows}
        n_utt, n_spk = write_split(rows, os.path.join(args.out_dir, split))
        print(f"{split}: {n_utt} utterances, {n_spk} speakers")

    # The split must be speaker-disjoint; stop early if it is not.
    names = list(speakers)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = speakers[a] & speakers[b]
            if shared:
                sys.exit(f"speakers in both {a} and {b}: {sorted(shared)[:5]}")


if __name__ == "__main__":
    main()
