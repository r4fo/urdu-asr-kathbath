"""Error analysis on the test predictions (after the shared normalization).

For each system listed below it counts:
  - loops: outputs where a 1-3 word phrase repeats 4 times in a row (Whisper hallucination)
  - latin: outputs that contain Latin letters (the model switched script)
  - substitution, deletion and insertion rates per reference word (one word alignment)
and, for some systems, which single letter is wrong in words that differ by exactly one letter.

Writes:
  results/error_analysis/summary.csv            one row per system
  results/error_analysis/letter_confusions.csv  system, reference letter, output letter, count

Reads only committed files (data/splits/test.csv, results/predictions/), so it can be re-run any time.
Usage (repo root):  python scripts/error_analysis.py
"""

import csv
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.normalize import normalize  # noqa: E402

SYSTEMS = ["whisper_small_zeroshot", "whisper_large_v3_zeroshot", "whisper_small_ft_beam5",
           "w2v2_xlsr_ft_greedy", "w2v2_xlsr_ft_kenlm"]
CONFUSION_SYSTEMS = ["whisper_small_ft_beam5", "w2v2_xlsr_ft_greedy", "w2v2_xlsr_ft_kenlm"]
OUT = Path("results/error_analysis")
LATIN = re.compile("[A-Za-z]")


def read(path, col):
    with open(path, encoding="utf-8") as fh:
        return {r["utt_id"]: (r[col] or "") for r in csv.DictReader(fh)}


def has_loop(words):
    for n in (1, 2, 3):
        for i in range(len(words) - 4 * n + 1):
            if all(words[i + k * n:i + (k + 1) * n] == words[i:i + n] for k in range(1, 4)):
                return True
    return False


def align(ref, hyp):
    """Levenshtein word alignment: returns (substituted pairs, deletions, insertions)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, subs, dels, ins = n, m, [], 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                subs.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dels, i = dels + 1, i - 1
        else:
            ins, j = ins + 1, j - 1
    return subs, dels, ins


def main():
    refs = {u: normalize(t).split() for u, t in read("data/splits/test.csv", "text").items()}
    n_words = sum(len(r) for r in refs.values())
    OUT.mkdir(parents=True, exist_ok=True)
    summary, confusions = [], []
    for system in SYSTEMS:
        raw = read(f"results/predictions/{system}.csv", "hyp")
        loops = latin = s = dl = ins = 0
        letters = Counter()
        for u, ref in refs.items():
            hyp = normalize(raw.get(u, "")).split()
            loops += has_loop(hyp)
            latin += bool(LATIN.search(raw.get(u, "")))
            subs, d, i = align(ref, hyp)
            s, dl, ins = s + len(subs), dl + d, ins + i
            for r, h in subs:
                diff = [(a, b) for a, b in zip(r, h) if a != b] if len(r) == len(h) else []
                if len(diff) == 1:
                    letters[diff[0]] += 1
        summary.append({"system": system, "utterances": len(refs), "loops": loops, "latin_outputs": latin,
                        "substitution_rate": round(s / n_words, 4), "deletion_rate": round(dl / n_words, 4),
                        "insertion_rate": round(ins / n_words, 4)})
        if system in CONFUSION_SYSTEMS:
            confusions += [{"system": system, "reference_letter": a, "output_letter": b, "count": c}
                           for (a, b), c in letters.most_common(15)]
    for name, rows in (("summary.csv", summary), ("letter_confusions.csv", confusions)):
        with (OUT / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    for r in summary:
        print(r)
    print(f"Wrote {OUT}/summary.csv and {OUT}/letter_confusions.csv ({n_words} reference words)")


if __name__ == "__main__":
    main()
