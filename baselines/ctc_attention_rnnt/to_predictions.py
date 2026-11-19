"""Convert an ESPnet decoding result into the shared prediction format.

ESPnet writes "<kaldi_id> <hypothesis>" lines to
exp/<asr_exp>/decode_*/<set>/text. This script removes the speaker prefix
added by prepare_data.py and writes results/predictions/<system>.csv with
columns utt_id,hyp (raw model output, not normalized).

Every utt_id of the split file must get a row; a missing one stops the script
so it is noticed (an empty hypothesis is kept as an empty string).

Usage:
  python baselines/ctc_attention_rnnt/to_predictions.py \
      --espnet-text <recipe>/exp/asr_ctc/decode_ctc_asr_model_valid.loss.best/test/text \
      --split data/splits/test.csv --out results/predictions/ctc.csv
"""

import argparse
import csv
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--espnet-text", required=True)
    parser.add_argument("--split", required=True, help="split CSV the decoding was run on")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    hyps = {}
    with open(args.espnet_text, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
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
