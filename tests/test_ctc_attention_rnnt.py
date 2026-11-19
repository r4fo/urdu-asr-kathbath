"""Tests for the ESPnet data conversion of the CTC / hybrid / RNN-T baselines
(baselines/ctc_attention_rnnt/). They need no ESPnet install and no GPU."""

import csv
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOLS = os.path.join(REPO_ROOT, "baselines", "ctc_attention_rnnt")
COLUMNS = ["utt_id", "path", "speaker_id", "gender", "duration", "text"]


def run(script, *args):
    return subprocess.run([sys.executable, os.path.join(TOOLS, script), *map(str, args)],
                          capture_output=True, text=True, encoding="utf-8")


def write_split(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)


@pytest.fixture
def splits(tmp_path):
    audio = tmp_path / "a.flac"
    audio.write_bytes(b"")  # only its existence is checked
    rel = os.path.relpath(audio, REPO_ROOT)
    d = tmp_path / "splits"
    d.mkdir()
    # Speaker 150 and 15 on purpose: plain string sorting would break Kathbath IDs.
    write_split(d / "train.csv", [
        ["u2-150-m", rel, "150", "Male", "1.0", "یہ  ایک جملہ ہے"],
        ["u1-15-f", rel, "15", "Female", "1.0", "دوسرا جملہ"],
    ])
    write_split(d / "dev.csv", [["u3-7-f", rel, "7", "Female", "1.0", "تیسرا"]])
    return d


def read_kaldi(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n").split(" ", 1) for line in f]


def test_prepare_data_writes_sorted_kaldi_dirs(splits, tmp_path):
    out = tmp_path / "data"
    r = run("prepare_data.py", "--splits-dir", splits, "--out-dir", out)
    assert r.returncode == 0, r.stderr

    text = read_kaldi(out / "train" / "text")
    ids = [k for k, _ in text]
    assert ids == sorted(ids)
    assert ids == ["S00015_u1-15-f", "S00150_u2-150-m"]
    assert text[1][1] == "یہ ایک جملہ ہے"  # spaces collapsed, letters unchanged

    utt2spk = read_kaldi(out / "train" / "utt2spk")
    spks = [s for _, s in utt2spk]
    assert spks == sorted(spks)  # Kaldi requirement: utt order = speaker order
    assert read_kaldi(out / "train" / "spk2utt") == [
        ["S00015", "S00015_u1-15-f"], ["S00150", "S00150_u2-150-m"]]
    assert not (out / "test").exists()  # test.csv is optional


def test_prepare_data_rejects_shared_speaker(splits, tmp_path):
    rows = list(csv.reader(open(splits / "train.csv", encoding="utf-8")))[1:]
    write_split(splits / "dev.csv", [rows[0]])
    r = run("prepare_data.py", "--splits-dir", splits, "--out-dir", tmp_path / "data")
    assert r.returncode != 0
    assert "speakers in both" in r.stderr


def test_to_predictions_round_trip(splits, tmp_path):
    hyp = tmp_path / "text"
    hyp.write_text("S00150_u2-150-m یہ ایک\nS00015_u1-15-f \n", encoding="utf-8")
    out = tmp_path / "pred.csv"
    r = run("to_predictions.py", "--espnet-text", hyp, "--split", splits / "train.csv", "--out", out)
    assert r.returncode == 0, r.stderr
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    # Same order and IDs as the split file; empty hypothesis kept as "".
    assert [(r["utt_id"], r["hyp"]) for r in rows] == [("u2-150-m", "یہ ایک"), ("u1-15-f", "")]


def test_to_predictions_fails_on_missing_utterance(splits, tmp_path):
    hyp = tmp_path / "text"
    hyp.write_text("S00150_u2-150-m یہ\n", encoding="utf-8")
    r = run("to_predictions.py", "--espnet-text", hyp, "--split", splits / "train.csv",
            "--out", tmp_path / "pred.csv")
    assert r.returncode != 0
