"""Make a tiny speaker-disjoint sample (default 40 train + 10 dev + 10 test clips) from
data/clean/metadata.csv, in the shared split format, to test the pipeline
before Member 2's real split exists. The sample goes to work/ (ignored by git)
and is never used for any reported number.

Usage:
  python baselines/ctc_attention_rnnt/make_sample_splits.py
"""

import argparse
import csv
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
COLUMNS = ["utt_id", "path", "speaker_id", "gender", "duration", "text"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default=os.path.join(REPO_ROOT, "data", "clean", "metadata.csv"))
    parser.add_argument("--out-dir", default=os.path.join(HERE, "work", "sample_splits"))
    parser.add_argument("--n-train", type=int, default=40)
    parser.add_argument("--n-dev", type=int, default=10)
    parser.add_argument("--n-test", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.metadata, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_spk = {}
    for r in rows:
        by_spk.setdefault(r["speaker_id"], []).append(r)
    speakers = sorted(by_spk, key=int)
    random.Random(args.seed).shuffle(speakers)

    # Dev and test from the first speakers, train from the rest, so no speaker is in two splits.
    def take(spks, n):
        out = []
        for s in spks:
            out += by_spk[s][: n - len(out)]
            if len(out) >= n:
                return out
        return out

    dev = take(speakers[:2], args.n_dev)
    test = take(speakers[2:4], args.n_test)
    train = take(speakers[4:], args.n_train)

    os.makedirs(args.out_dir, exist_ok=True)
    for name, part in (("train", train), ("dev", dev), ("test", test)):
        with open(os.path.join(args.out_dir, f"{name}.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(part)
        print(f"{name}: {len(part)} clips -> {args.out_dir}/{name}.csv")


if __name__ == "__main__":
    main()
