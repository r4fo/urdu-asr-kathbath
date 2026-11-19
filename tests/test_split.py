"""Tests for scripts/split.py on fake metadata shaped like the real data (no real data needed)."""
import csv
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import split as sp  # noqa: E402

SENTENCES = ["میں گھر گیا", "وہ آیا", "کیا آپ آئیں گے", "آج موسم اچھا ہے", "یہ کتاب میری ہے"]


def fake_metadata(path, seed=0):
    """47 speakers like the real data: 21 Female / 26 Male, most ~0.84 h, some with very few clips."""
    rng = random.Random(seed)
    rows = []
    for i in range(47):
        gender = "Female" if i < 21 else "Male"
        n_clips = rng.choice([5, 12, 40]) if i % 6 == 0 else rng.choice([300, 400, 500, 550])
        for c in range(n_clips):
            text = f"جملہ {i} {c}"
            if c == 0:
                text = SENTENCES[i % len(SENTENCES)]  # some sentences shared across speakers
            rows.append([f"{i}_{c}", f"data/clean/{i}_{c}.flac", str(i), gender, f"{rng.uniform(4, 8):.2f}", text])
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(sp.COLUMNS + ["source_split"])
        w.writerows([r + ["train"] for r in rows])
    return rows


@pytest.fixture
def run(tmp_path):
    meta = tmp_path / "metadata.csv"
    rows = fake_metadata(meta)
    out = tmp_path / "splits"
    splits = sp.main(["--metadata", str(meta), "--out_dir", str(out)])
    return rows, splits, out


def read(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def speakers(rs):
    return {r["speaker_id"] for r in rs}


def test_no_speaker_in_two_splits(run):
    _, splits, _ = run
    assert not speakers(splits["train"]) & speakers(splits["dev"])
    assert not speakers(splits["train"]) & speakers(splits["test"])
    assert not speakers(splits["dev"]) & speakers(splits["test"])


def test_every_clip_in_exactly_one_split(run):
    rows, splits, _ = run
    ids = [r["utt_id"] for v in splits.values() for r in v]
    assert len(ids) == len(set(ids)) == len(rows)


def test_speaker_counts_and_gender_balance(run):
    _, splits, _ = run
    for name, female, male in (("test", 4, 5), ("dev", 3, 3)):
        g = {}
        for r in splits[name]:
            g[r["speaker_id"]] = r["gender"]
        assert list(g.values()).count("Female") == female
        assert list(g.values()).count("Male") == male


def test_hour_shares_in_range(run):
    _, splits, _ = run
    h = {k: sum(float(r["duration"]) for r in v) for k, v in splits.items()}
    total = sum(h.values())
    assert 0.17 <= h["test"] / total <= 0.19
    assert 0.10 <= h["dev"] / total <= 0.12


def test_small_speakers_stay_in_train(run):
    rows, splits, _ = run
    clips = {}
    for r in rows:
        clips[r[2]] = clips.get(r[2], 0) + 1
    for name in ("test", "dev"):
        assert all(clips[s] >= 50 for s in speakers(splits[name]))


def test_output_files_and_columns(run):
    _, splits, out = run
    for name in ("train", "dev", "test"):
        rows = read(out / f"{name}.csv")
        assert list(rows[0]) == sp.COLUMNS
        assert len(rows) == len(splits[name])
    stats = {r["split"]: r for r in read(out / "split_stats.csv")}
    assert set(stats) == {"train", "dev", "test"}
    assert stats["test"]["speakers"] == "9" and stats["dev"]["speakers"] == "6"
    assert {"speakers_Female", "speakers_Male", "utterances", "hours", "pct_hours"} <= set(stats["train"])


def test_overlap_includes_shared_sentences(run):
    _, splits, out = run
    overlap = {r["utt_id"] for r in read(out / "test_overlap.csv")}
    train_texts = {r["text"] for r in splits["train"]}
    expected = {r["utt_id"] for r in splits["test"] if r["text"] in train_texts}
    assert overlap == expected and overlap


def test_overlap_uses_normalized_text():
    splits = {"train": [{"utt_id": "a", "text": "آپ کیسے ہیں"}],
              "test": [{"utt_id": "b", "text": "اپ کیسے ہیں"}, {"utt_id": "c", "text": "نیا جملہ"}]}
    assert [r["utt_id"] for r in sp.test_overlap(splits)] == ["b"]


def test_same_seed_same_split(tmp_path):
    meta = tmp_path / "m.csv"
    fake_metadata(meta)
    a = sp.main(["--metadata", str(meta), "--out_dir", str(tmp_path / "a")])
    b = sp.main(["--metadata", str(meta), "--out_dir", str(tmp_path / "b")])
    assert speakers(a["test"]) == speakers(b["test"]) and speakers(a["dev"]) == speakers(b["dev"])
    c = sp.main(["--metadata", str(meta), "--out_dir", str(tmp_path / "c"), "--seed", "7"])
    assert speakers(a["test"]) != speakers(c["test"])


def test_per_gender_counts():
    assert sp.per_gender(9, {"Female": list(range(21)), "Male": list(range(26))}) == {"Female": 4, "Male": 5}
    assert sp.per_gender(6, {"Female": list(range(21)), "Male": list(range(26))}) == {"Female": 3, "Male": 3}


def test_absolute_path_fails(tmp_path):
    meta = tmp_path / "m.csv"
    meta.write_text("utt_id,path,speaker_id,gender,duration,text\n"
                    "u1,/home/x/a.flac,s1,Male,1.0,میں\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="relative"):
        sp.main(["--metadata", str(meta), "--out_dir", str(tmp_path / "o")])
    meta.write_text("utt_id,path,speaker_id,gender,duration,text\n"
                    "u1,C:\\Users\\x\\a.flac,s1,Male,1.0,میں\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="relative"):
        sp.main(["--metadata", str(meta), "--out_dir", str(tmp_path / "o")])


def test_missing_column_fails(tmp_path):
    meta = tmp_path / "m.csv"
    meta.write_text("utt_id,path\nu1,a.flac\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="missing columns"):
        sp.main(["--metadata", str(meta), "--out_dir", str(tmp_path / "o")])
