"""Turn the shared split CSVs into Kaldi data folders (Member 3, GMM-HMM baseline).

Input:  <splits-dir>/train.csv, dev.csv, test.csv with columns
        utt_id,path,speaker_id,gender,duration,text
Output: <out-dir>/<split>/wav.scp, text, utt2spk, spk2utt

- Utterance IDs become "S<5-digit speaker>_<utt_id>": Kaldi needs the speaker ID first so that
  sorting by utterance also sorts by speaker. baselines/gmm_hmm/to_predictions.py removes it again.
- wav.scp reads the FLAC files through `flac -c -d -s`, because Kaldi does not read FLAC itself.
- The Kaldi `text` file is normalized with src/normalize.py (the team's one normalization), so the
  training text, the lexicon and the LM do not contain punctuation or letter variants. The split CSVs
  and the prediction files are not changed; evaluate.py normalizes references and predictions itself.
- The script stops if a speaker appears in two splits.

Usage (from the repo root):
  python scripts/make_kaldi_dirs.py --out-dir baselines/gmm_hmm/work/data
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
from src.normalize import normalize  # noqa: E402

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


def write_split(rows, out_dir, repo_root, check_audio=True):
    os.makedirs(out_dir, exist_ok=True)
    entries, skipped = [], 0
    for row in rows:
        text = normalize(row["text"])
        if not text:
            skipped += 1
            print(f"skip {row['utt_id']}: empty text after normalization", file=sys.stderr)
            continue
        audio = os.path.join(repo_root, row["path"])
        if check_audio and not os.path.isfile(audio):
            sys.exit(f"audio not found: {audio}")
        spk = f"S{int(row['speaker_id']):05d}"
        entries.append((kaldi_id(row["speaker_id"], row["utt_id"]), audio, text, spk))
    entries.sort()  # byte order, the same as LC_ALL=C sort

    ids = [e[0] for e in entries]
    if len(ids) != len(set(ids)):
        sys.exit(f"{out_dir}: duplicate utt_id")

    spk2utt = defaultdict(list)
    with open(os.path.join(out_dir, "wav.scp"), "w", encoding="utf-8") as wav, \
         open(os.path.join(out_dir, "text"), "w", encoding="utf-8") as txt, \
         open(os.path.join(out_dir, "utt2spk"), "w", encoding="utf-8") as u2s:
        for uid, audio, text, spk in entries:
            wav.write(f"{uid} flac -c -d -s {audio} |\n")
            txt.write(f"{uid} {text}\n")
            u2s.write(f"{uid} {spk}\n")
            spk2utt[spk].append(uid)
    with open(os.path.join(out_dir, "spk2utt"), "w", encoding="utf-8") as s2u:
        for spk in sorted(spk2utt):
            s2u.write(f"{spk} {' '.join(spk2utt[spk])}\n")
    return len(entries), len(spk2utt), skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--splits-dir", default=os.path.join(REPO_ROOT, "data", "splits"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repo-root", default=REPO_ROOT,
                        help="folder the 'path' column is relative to (the repo root)")
    args = parser.parse_args()

    data, owner = {}, {}
    for split in SPLITS:
        p = os.path.join(args.splits_dir, f"{split}.csv")
        if not os.path.isfile(p):
            if split == "test":
                continue  # test is optional until the final run
            sys.exit(f"missing {p}")
        data[split] = read_split(p)
        for row in data[split]:
            prev = owner.setdefault(row["speaker_id"], split)
            if prev != split:
                sys.exit(f"speaker {row['speaker_id']} is in both {prev} and {split}")

    for split, rows in data.items():
        n, n_spk, skipped = write_split(rows, os.path.join(args.out_dir, split), args.repo_root)
        print(f"{split}: {n} utterances, {n_spk} speakers, {skipped} skipped")


if __name__ == "__main__":
    main()
