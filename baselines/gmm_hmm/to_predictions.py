"""Convert a Kaldi decoding result into the shared prediction format.

Kaldi writes "<kaldi_id> <hypothesis>" lines (exp/<model>/decode_<set>/scoring_kaldi/penalty_0.0/<LMWT>.txt).
This script removes the speaker prefix added by scripts/make_kaldi_dirs.py and writes
results/predictions/<system>.csv with columns utt_id,hyp. The hypothesis is the raw Kaldi output
(it is already in normalized form because the model was trained on normalized text); do not edit it.

Every utt_id of the split file must get a row; a missing one stops the script. An utterance for which
the decoder produced no words is kept with an empty hypothesis.

Usage:
  python baselines/gmm_hmm/to_predictions.py --kaldi-text <hyp.txt> \
      --split data/splits/test.csv --out results/predictions/gmm_hmm.csv
"""
import argparse
import csv
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--kaldi-text", required=True)
    parser.add_argument("--split", required=True, help="split CSV the decoding was run on")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    hyps = {}
    with open(args.kaldi_text, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            kid, _, hyp = line.partition(" ")
            utt_id = kid.split("_", 1)[1]  # "S00015_<utt_id>" -> "<utt_id>"
            hyps[utt_id] = hyp.strip()

    with open(args.split, encoding="utf-8", newline="") as f:
        utt_ids = [r["utt_id"] for r in csv.DictReader(f)]
    missing = [u for u in utt_ids if u not in hyps]
    if missing:
        sys.exit(f"{len(missing)} utterances have no hypothesis, e.g. {missing[:3]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "hyp"])
        for u in utt_ids:
            w.writerow([u, hyps[u]])
    print(f"wrote {len(utt_ids)} rows to {args.out}")


if __name__ == "__main__":
    main()
