"""Tests for the GMM-HMM baseline helpers (scripts/make_kaldi_dirs.py and baselines/gmm_hmm/).
They need no Kaldi install."""
import csv
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GMM = os.path.join(REPO_ROOT, "baselines", "gmm_hmm")
COLUMNS = ["utt_id", "path", "speaker_id", "gender", "duration", "text"]


def run(script, *args):
    return subprocess.run([sys.executable, script, *map(str, args)],
                          capture_output=True, text=True, encoding="utf-8", cwd=REPO_ROOT)


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
    # Speaker 150 and 15 on purpose: plain string sorting would break the speaker order.
    write_split(d / "train.csv", [
        ["u2-150-m", rel, "150", "Male", "1.0", "یہ  ایک جملہ، ہے۔"],
        ["u1-15-f", rel, "15", "Female", "2.0", "دوسرا جملہ"],
    ])
    write_split(d / "dev.csv", [["u3-7-f", rel, "7", "Female", "1.5", "تیسرا جملہ نیا"]])
    return d


def test_kaldi_dirs(splits, tmp_path):
    out = tmp_path / "data"
    r = run(os.path.join(REPO_ROOT, "scripts", "make_kaldi_dirs.py"), "--splits-dir", splits, "--out-dir", out)
    assert r.returncode == 0, r.stderr
    text = (out / "train" / "text").read_text(encoding="utf-8").splitlines()
    assert [t.split()[0] for t in text] == ["S00015_u1-15-f", "S00150_u2-150-m"]
    assert text[1] == "S00150_u2-150-m یہ ایک جملہ ہے"  # normalized: punctuation removed, spaces collapsed
    assert "flac -c -d -s" in (out / "train" / "wav.scp").read_text(encoding="utf-8")
    assert (out / "train" / "spk2utt").read_text(encoding="utf-8").startswith("S00015 S00015_u1-15-f")


def test_shared_speaker_is_rejected(splits, tmp_path):
    rows = list(csv.reader(open(splits / "dev.csv", encoding="utf-8")))
    rows[1][2] = "15"
    write_split(splits / "dev.csv", rows[1:])
    r = run(os.path.join(REPO_ROOT, "scripts", "make_kaldi_dirs.py"), "--splits-dir", splits,
            "--out-dir", tmp_path / "data")
    assert r.returncode != 0 and "speaker 15" in r.stderr


def test_dict_and_oov(splits, tmp_path):
    out = tmp_path / "data"
    run(os.path.join(REPO_ROOT, "scripts", "make_kaldi_dirs.py"), "--splits-dir", splits, "--out-dir", out)
    r = run(os.path.join(GMM, "make_dict.py"), "--train-text", out / "train" / "text",
            "--dict-dir", tmp_path / "dict", "--check", out / "dev" / "text")
    assert r.returncode == 0, r.stderr
    lex = (tmp_path / "dict" / "lexicon.txt").read_text(encoding="utf-8").splitlines()
    assert "<unk> spn" in lex and "!SIL sil" in lex
    assert "ایک ا ی ک" in lex
    assert "OOV rate 66.67% (2 of 3 words)" in r.stdout  # only "جملہ" of the 3 dev words is in train


def test_to_predictions(splits, tmp_path):
    hyp = tmp_path / "hyp.txt"
    hyp.write_text("S00007_u3-7-f تیسرا جملہ\n", encoding="utf-8")
    out = tmp_path / "pred.csv"
    r = run(os.path.join(GMM, "to_predictions.py"), "--kaldi-text", hyp, "--split", splits / "dev.csv", "--out", out)
    assert r.returncode == 0, r.stderr
    assert list(csv.reader(open(out, encoding="utf-8"))) == [["utt_id", "hyp"], ["u3-7-f", "تیسرا جملہ"]]


def test_to_predictions_stops_on_missing_utterance(splits, tmp_path):
    hyp = tmp_path / "hyp.txt"
    hyp.write_text("S00015_u1-15-f کچھ\n", encoding="utf-8")
    r = run(os.path.join(GMM, "to_predictions.py"), "--kaldi-text", hyp, "--split", splits / "dev.csv",
            "--out", tmp_path / "pred.csv")
    assert r.returncode != 0 and "no hypothesis" in r.stderr


def test_empty_hypothesis_is_kept(splits, tmp_path):
    hyp = tmp_path / "hyp.txt"
    hyp.write_text("S00007_u3-7-f\n", encoding="utf-8")  # Kaldi writes just the ID for an empty result
    out = tmp_path / "pred.csv"
    r = run(os.path.join(GMM, "to_predictions.py"), "--kaldi-text", hyp, "--split", splits / "dev.csv", "--out", out)
    assert r.returncode == 0, r.stderr
    assert list(csv.reader(open(out, encoding="utf-8")))[1] == ["u3-7-f", ""]
