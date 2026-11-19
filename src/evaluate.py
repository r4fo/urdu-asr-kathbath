"""Score one prediction file against a split file: WER, CER, and a bootstrap 95% CI on WER.

Usage:
    python src/evaluate.py --split data/splits/test.csv --pred results/predictions/<system>.csv
    python src/evaluate.py --split data/splits/test.csv --pred results/predictions/*.csv   # all systems
    # Normalization comparison (extra table only; the main results use full normalization):
    python src/evaluate.py --split data/splits/test.csv --pred results/predictions/<system>.csv --no-normalize
    python src/evaluate.py --split data/splits/test.csv --pred results/predictions/<system>.csv --no-digits
    # WER on test clips whose sentence does not appear in train (for LM systems):
    python src/evaluate.py --split data/splits/test.csv --pred results/predictions/<system>.csv \
        --exclude data/splits/test_overlap.csv

Appends one row to results/results.csv: system,split,n_utts,wer,cer,wer_ci_low,wer_ci_high
The split column says how the row was scored:
    test                   full normalization (src/normalize.py), used for the main results table
    test_nodigits          normalization without digits -> words (--no-digits)
    test_nonorm            raw text, no normalization at all (--no-normalize)
    test_nonoverlap[...]   only test clips whose sentence is not in train (--exclude)
WER and CER are fractions (0.25 = 25%). Both are corpus-level: total errors / total reference words (or characters).
CER counts spaces as characters.
"""
import argparse
import csv
import sys
from pathlib import Path

import jiwer
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.normalize import normalize  # noqa: E402

RESULT_COLUMNS = ["system", "split", "n_utts", "wer", "cer", "wer_ci_low", "wer_ci_high"]
N_BOOTSTRAP = 1000
SEED = 42


def read_csv(path):
    # utf-8-sig also reads files saved with a BOM (Excel, pandas on Windows)
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_pairs(split_path, pred_path, exclude_path=None, mode="full"):
    """Return (utt_ids, refs, hyps) in split-file order. Fails if any utt_id has no prediction.

    mode: "full" = normalize(), "nodigits" = normalize() without digits -> words, "none" = raw text.
    """
    if mode == "none":
        norm = lambda s: (s or "").strip()  # noqa: E731
    elif mode == "nodigits":
        norm = lambda s: normalize(s, digits=False)  # noqa: E731
    elif mode == "full":
        norm = normalize
    else:
        raise ValueError(f"unknown mode {mode}")
    split_rows = read_csv(split_path)
    pred_rows = read_csv(pred_path)
    if pred_rows and set(pred_rows[0]) != {"utt_id", "hyp"}:
        raise SystemExit(f"{pred_path}: header must be utt_id,hyp, got {list(pred_rows[0])}")
    preds = {}
    for r in pred_rows:
        if r["utt_id"] in preds:
            raise SystemExit(f"{pred_path}: duplicate utt_id {r['utt_id']}")
        preds[r["utt_id"]] = r["hyp"] or ""

    excluded = {r["utt_id"] for r in read_csv(exclude_path)} if exclude_path else set()
    split_rows = [r for r in split_rows if r["utt_id"] not in excluded]

    missing = [r["utt_id"] for r in split_rows if r["utt_id"] not in preds]
    if missing:
        raise SystemExit(f"{pred_path}: {len(missing)} utt_ids have no prediction, e.g. {missing[:5]}")
    extra = set(preds) - {r["utt_id"] for r in read_csv(split_path)}
    if extra:
        print(f"WARNING: {len(extra)} predictions are not in {split_path} and are ignored, e.g. {sorted(extra)[:5]}")

    refs = [norm(r["text"]) for r in split_rows]
    empty = [r["utt_id"] for r, ref in zip(split_rows, refs) if not ref]
    if empty:
        raise SystemExit(f"{len(empty)} references are empty after normalization, e.g. {empty[:5]}")
    hyps = [norm(preds[r["utt_id"]]) for r in split_rows]
    return [r["utt_id"] for r in split_rows], refs, hyps


def per_utt_word_errors(refs, hyps):
    """Word errors (S + D + I) and reference word count for every utterance."""
    out = jiwer.process_words(refs, hyps)
    errors = []
    for chunks in out.alignments:
        e = 0
        for c in chunks:
            if c.type in ("substitute", "delete"):
                e += c.ref_end_idx - c.ref_start_idx
            elif c.type == "insert":
                e += c.hyp_end_idx - c.hyp_start_idx
        errors.append(e)
    ref_words = [len(r.split()) for r in refs]
    return np.array(errors), np.array(ref_words), out.wer


def bootstrap_wer_ci(errors, ref_words, n=N_BOOTSTRAP, seed=SEED):
    """95% CI of corpus WER: resample utterances with replacement n times."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(errors), size=(n, len(errors)))
    wers = errors[idx].sum(axis=1) / ref_words[idx].sum(axis=1)
    return float(np.percentile(wers, 2.5)), float(np.percentile(wers, 97.5))


def score(refs, hyps):
    errors, ref_words, wer = per_utt_word_errors(refs, hyps)
    low, high = bootstrap_wer_ci(errors, ref_words)
    return {"wer": wer, "cer": jiwer.cer(refs, hyps), "wer_ci_low": low, "wer_ci_high": high}


def append_result(results_path, row):
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    new = not results_path.exists()
    with open(results_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, help="split CSV with utt_id and text")
    p.add_argument("--pred", required=True, nargs="+", help="one or more prediction CSVs with utt_id,hyp")
    p.add_argument("--system", help="only with one --pred; defaults to the prediction file name without .csv")
    p.add_argument("--exclude", help="CSV with a utt_id column; those clips are not scored")
    p.add_argument("--results", default=str(REPO / "results/results.csv"))
    g = p.add_mutually_exclusive_group()
    g.add_argument("--no-normalize", action="store_true", help="score raw text, no normalization at all")
    g.add_argument("--no-digits", action="store_true", help="full normalization but skip digits -> words")
    args = p.parse_args(argv)
    mode = "none" if args.no_normalize else "nodigits" if args.no_digits else "full"

    if args.system and len(args.pred) > 1:
        raise SystemExit("--system can only be used with a single --pred file")
    split = (Path(args.split).stem + ("_nonoverlap" if args.exclude else "")
             + {"full": "", "nodigits": "_nodigits", "none": "_nonorm"}[mode])
    rows = []
    for pred in args.pred:
        system = args.system or Path(pred).stem
        _, refs, hyps = load_pairs(args.split, pred, args.exclude, mode)
        s = score(refs, hyps)
        row = {"system": system, "split": split, "n_utts": len(refs),
               **{k: round(v, 6) for k, v in s.items()}}
        append_result(args.results, row)
        print(f"{system} on {split} ({len(refs)} utts): WER {s['wer']:.2%} "
              f"[{s['wer_ci_low']:.2%}, {s['wer_ci_high']:.2%}]  CER {s['cer']:.2%}")
        rows.append(row)
    return rows[0] if len(rows) == 1 else rows


if __name__ == "__main__":
    main()
