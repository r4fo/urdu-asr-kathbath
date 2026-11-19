"""Make the train / dev / test split files from the clean metadata, by speaker.

Usage:
    python scripts/split.py

The Hugging Face version of Kathbath has no Test-Unknown split, and every valid speaker is also a
train speaker, so we split all speakers ourselves. Every speaker goes whole to one split.

Rule (seed 42):
1. Test gets 9 speakers and dev 6, divided between genders in proportion to the speaker counts
   (47 speakers, 21 F / 26 M -> test 4 F + 5 M, dev 3 F + 3 M).
2. Only speakers with at least 50 clips can go to test or dev. Speakers with very few clips
   would add a speaker to the count but almost no audio, so they always go to train.
3. Speakers are drawn at random with seed 42. We keep the first draw where test has 17-19% of all
   hours and dev 10-12%, so a lucky draw of very large or very small speakers cannot skew the sizes.
   With 9 speakers of about 0.84 h each, test cannot be much smaller than about 18%.

Writes to data/splits/:
    train.csv, dev.csv, test.csv   utt_id,path,speaker_id,gender,duration,text (text not normalized)
    split_stats.csv                speakers (per gender), utterances, hours, % of hours per split
    test_overlap.csv               test clips whose normalized sentence also appears in train
"""
import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.normalize import normalize  # noqa: E402

COLUMNS = ["utt_id", "path", "speaker_id", "gender", "duration", "text"]
SEED = 42
MAX_DRAWS = 1_000_000


def read_metadata(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    missing = set(COLUMNS) - set(rows[0] if rows else [])
    if missing:
        raise SystemExit(f"{path} is missing columns: {sorted(missing)}")
    # Catch both Linux (/home/...) and Windows (C:\...) absolute paths on any OS.
    absolute = [r["path"] for r in rows
                if PurePosixPath(r["path"]).is_absolute() or PureWindowsPath(r["path"]).is_absolute()]
    if absolute:
        raise SystemExit(f"Paths must be relative to the repo root, e.g. {absolute[0]}")
    missing_audio = [r["path"] for r in rows if not (REPO / r["path"]).exists()]
    if missing_audio:
        print(f"WARNING: {len(missing_audio)} audio files not found under the repo, e.g. {missing_audio[0]}")
    ids = [r["utt_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{path} has duplicate utt_ids")
    genders = defaultdict(set)
    for r in rows:
        genders[r["speaker_id"]].add(r["gender"])
    mixed = [s for s, g in genders.items() if len(g) > 1]
    if mixed:
        raise SystemExit(f"Speakers with more than one gender value: {mixed[:5]}")
    return rows


def per_gender(total, speakers_by_gender):
    """Split `total` speakers between genders in proportion to their speaker counts (largest remainder)."""
    n_all = sum(len(s) for s in speakers_by_gender.values())
    exact = {g: total * len(s) / n_all for g, s in speakers_by_gender.items()}
    out = {g: int(x) for g, x in exact.items()}
    for g in sorted(exact, key=lambda g: (-(exact[g] - out[g]), g))[:total - sum(out.values())]:
        out[g] += 1
    return out


def speaker_table(rows):
    hours, clips, gender = defaultdict(float), defaultdict(int), {}
    for r in rows:
        s = r["speaker_id"]
        hours[s] += float(r["duration"]) / 3600
        clips[s] += 1
        gender[s] = r["gender"]
    return hours, clips, gender


def choose_speakers(rows, n_test=9, n_dev=6, min_clips=50, test_range=(0.17, 0.19),
                    dev_range=(0.10, 0.12), seed=SEED):
    """Return (test_speakers, dev_speakers, draw_number)."""
    hours, clips, gender = speaker_table(rows)
    total = sum(hours.values())
    all_by_g, eligible_by_g = defaultdict(list), defaultdict(list)
    for s in sorted(gender, key=lambda s: (len(s), s)):
        all_by_g[gender[s]].append(s)
        if clips[s] >= min_clips:
            eligible_by_g[gender[s]].append(s)
    t_counts, d_counts = per_gender(n_test, all_by_g), per_gender(n_dev, all_by_g)
    for g in all_by_g:
        if t_counts[g] + d_counts[g] > len(eligible_by_g[g]) - 1:
            raise SystemExit(f"Not enough {g} speakers with >= {min_clips} clips for test and dev")

    rng = random.Random(seed)
    for draw in range(1, MAX_DRAWS + 1):
        test, dev = [], []
        for g in sorted(eligible_by_g):
            s = list(eligible_by_g[g])
            rng.shuffle(s)
            test += s[:t_counts[g]]
            dev += s[t_counts[g]:t_counts[g] + d_counts[g]]
        t_frac = sum(hours[s] for s in test) / total
        d_frac = sum(hours[s] for s in dev) / total
        if test_range[0] <= t_frac <= test_range[1] and dev_range[0] <= d_frac <= dev_range[1]:
            return set(test), set(dev), draw
    raise SystemExit("No draw met the hour ranges; widen --test_range / --dev_range")


def assign(rows, test_speakers, dev_speakers):
    by_id = lambda rs: sorted(rs, key=lambda r: r["utt_id"])  # noqa: E731
    return {
        "train": by_id([r for r in rows if r["speaker_id"] not in test_speakers | dev_speakers]),
        "dev": by_id([r for r in rows if r["speaker_id"] in dev_speakers]),
        "test": by_id([r for r in rows if r["speaker_id"] in test_speakers]),
    }


def stats(splits):
    out = []
    genders = sorted({r["gender"] for rs in splits.values() for r in rs})
    total = sum(float(r["duration"]) for rs in splits.values() for r in rs)
    for name, rs in splits.items():
        row = {"split": name, "speakers": len({r["speaker_id"] for r in rs})}
        for g in genders:
            row[f"speakers_{g}"] = len({r["speaker_id"] for r in rs if r["gender"] == g})
        sec = sum(float(r["duration"]) for r in rs)
        row["utterances"] = len(rs)
        row["hours"] = round(sec / 3600, 3)
        row["pct_hours"] = round(100 * sec / total, 1)
        out.append(row)
    return out


def test_overlap(splits):
    """Test clips whose normalized sentence also appears in train (includes sentences read twice)."""
    train_sentences = {normalize(r["text"]) for r in splits["train"]}
    return [{"utt_id": r["utt_id"], "text": r["text"]}
            for r in splits["test"] if normalize(r["text"]) in train_sentences]


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", default=str(REPO / "data/clean/metadata.csv"))
    p.add_argument("--out_dir", default=str(REPO / "data/splits"))
    p.add_argument("--n_test", type=int, default=9, help="test speakers")
    p.add_argument("--n_dev", type=int, default=6, help="dev speakers")
    p.add_argument("--min_clips", type=int, default=50, help="minimum clips for a test or dev speaker")
    p.add_argument("--test_range", type=float, nargs=2, default=[0.17, 0.19], help="allowed share of hours")
    p.add_argument("--dev_range", type=float, nargs=2, default=[0.10, 0.12], help="allowed share of hours")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)

    rows = read_metadata(args.metadata)
    test_spk, dev_spk, draw = choose_speakers(rows, args.n_test, args.n_dev, args.min_clips,
                                              tuple(args.test_range), tuple(args.dev_range), args.seed)
    splits = assign(rows, test_spk, dev_spk)

    out = Path(args.out_dir)
    for name, rs in splits.items():
        write_csv(out / f"{name}.csv", rs, COLUMNS)
    st = stats(splits)
    write_csv(out / "split_stats.csv", st, list(st[0]))
    overlap = test_overlap(splits)
    write_csv(out / "test_overlap.csv", overlap, ["utt_id", "text"])

    print(f"Seed {args.seed}, accepted draw {draw}")
    for name, spk in (("test", test_spk), ("dev", dev_spk)):
        print(f"{name} speakers: {', '.join(sorted(spk, key=lambda s: (len(s), s)))}")
    for row in st:
        print(row)
    print(f"Test clips whose sentence also appears in train: {len(overlap)} of {len(splits['test'])}")
    return splits


if __name__ == "__main__":
    main()
