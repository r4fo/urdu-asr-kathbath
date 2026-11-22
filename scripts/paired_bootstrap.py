"""Paired bootstrap for the key comparisons on the test set.

For each pair (system A, system B) it resamples the 3,031 test utterances with replacement (the same
resample for both systems), 1000 times with seed 42, and reports the WER difference A - B in percentage
points with a 95% percentile interval. An interval that does not contain 0 means the difference is
unlikely to come from which utterances happen to be in the test set.

Reads data/splits/test.csv and results/predictions/; writes results/paired_bootstrap.csv.
Usage (repo root):  python scripts/paired_bootstrap.py
"""

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.normalize import normalize  # noqa: E402

PAIRS = [
    ("whisper_small_ft_beam5", "whisper_small_zeroshot", "fine-tuning Whisper-small"),
    ("whisper_small_ft_beam5", "whisper_large_v3_zeroshot", "fine-tuned small vs zero-shot large-v3"),
    ("w2v2_xlsr_ft_kenlm", "w2v2_xlsr_ft_greedy", "adding the KenLM"),
    ("w2v2_xlsr_ft_kenlm", "whisper_small_ft_beam5", "best wav2vec2 vs best Whisper"),
    ("whisper_small_ft_beam5", "whisper_small_ft_beam1", "beam 5 vs beam 1"),
]
N_BOOT, SEED = 1000, 42
OUT = Path("results/paired_bootstrap.csv")


def word_errors(ref, hyp):
    """Word-level Levenshtein distance."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def per_utt_errors(system, refs):
    with open(f"results/predictions/{system}.csv", encoding="utf-8") as fh:
        hyps = {r["utt_id"]: normalize(r["hyp"] or "").split() for r in csv.DictReader(fh)}
    return [word_errors(refs[u], hyps.get(u, [])) for u in refs]


def main():
    with open("data/splits/test.csv", encoding="utf-8") as fh:
        refs = {r["utt_id"]: normalize(r["text"]).split() for r in csv.DictReader(fh)}
    n_words = [len(r) for r in refs.values()]
    cache, rows = {}, []
    for a, b, label in PAIRS:
        for s in (a, b):
            if s not in cache:
                cache[s] = per_utt_errors(s, refs)
        ea, eb = cache[a], cache[b]
        diff = 100 * (sum(ea) - sum(eb)) / sum(n_words)
        rng = random.Random(SEED)
        n = len(n_words)
        boots = []
        for _ in range(N_BOOT):
            idx = [rng.randrange(n) for _ in range(n)]
            words = sum(n_words[i] for i in idx)
            boots.append(100 * (sum(ea[i] for i in idx) - sum(eb[i] for i in idx)) / words)
        boots.sort()
        lo, hi = boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT) - 1]
        rows.append({"system_a": a, "system_b": b, "comparison": label,
                     "wer_diff_points": round(diff, 2), "ci_low": round(lo, 2), "ci_high": round(hi, 2),
                     "significant": not (lo <= 0 <= hi)})
        print(rows[-1])
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
