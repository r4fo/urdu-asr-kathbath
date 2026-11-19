"""Tests for src/evaluate.py with hand-checked WER and CER."""
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import evaluate as ev  # noqa: E402


def write(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def split_file(tmp_path, texts):
    rows = [[f"u{i}", f"a/u{i}.flac", "s1", "male", "1.0", t] for i, t in enumerate(texts)]
    return write(tmp_path / "test.csv", ["utt_id", "path", "speaker_id", "gender", "duration", "text"], rows)


def pred_file(tmp_path, hyps, name="sys_a.csv"):
    return write(tmp_path / name, ["utt_id", "hyp"], [[f"u{i}", h] for i, h in enumerate(hyps)])


def test_perfect_match_is_zero():
    s = ev.score(["میں گھر گیا"], ["میں گھر گیا"])
    assert s["wer"] == 0 and s["cer"] == 0 and s["wer_ci_low"] == s["wer_ci_high"] == 0


def test_one_substitution_hand_checked():
    # ref 3 words, 1 substituted -> WER 1/3. Characters: "میں گھر گیا" = 11, "گیا"->"آیا" changes 1 char -> CER 1/11.
    s = ev.score(["میں گھر گیا"], ["میں گھر آیا"])
    assert s["wer"] == pytest.approx(1 / 3)
    assert s["cer"] == pytest.approx(1 / 11)


def test_deletion_and_insertion_hand_checked():
    # utt 1: ref "a b c d", hyp "a b d"      -> 1 deletion
    # utt 2: ref "x y",     hyp "x y z w"    -> 2 insertions
    # corpus WER = 3 errors / 6 reference words = 0.5
    s = ev.score(["ایک دو تین چار", "پانچ چھ"], ["ایک دو چار", "پانچ چھ سات آٹھ"])
    assert s["wer"] == pytest.approx(0.5)


def test_corpus_wer_not_mean_of_utterance_wers():
    # utt 1: 1 error / 1 word (WER 1.0); utt 2: 0 errors / 4 words. Corpus = 1/5, mean would be 0.5.
    s = ev.score(["ایک", "دو تین چار پانچ"], ["سات", "دو تین چار پانچ"])
    assert s["wer"] == pytest.approx(0.2)


def test_empty_hypothesis_counts_all_deletions():
    s = ev.score(["میں گھر گیا"], [""])
    assert s["wer"] == pytest.approx(1.0) and s["cer"] == pytest.approx(1.0)


def test_per_utterance_errors_add_up_to_jiwer_wer():
    refs = ["ایک دو تین", "چار پانچ چھ سات", "آٹھ"]
    hyps = ["ایک تین", "چار پانچ چھ سات نو دس", "نو"]
    errors, ref_words, wer = ev.per_utt_word_errors(refs, hyps)
    assert list(errors) == [1, 2, 1] and list(ref_words) == [3, 4, 1]
    assert errors.sum() / ref_words.sum() == pytest.approx(wer) == pytest.approx(0.5)


def test_bootstrap_ci_contains_wer_and_is_deterministic():
    refs = ["ایک دو تین", "چار پانچ", "چھ سات آٹھ نو", "دس"] * 10
    hyps = ["ایک دو تین", "چار", "چھ سات آٹھ دس", "گیارہ"] * 10
    a, b = ev.score(refs, hyps), ev.score(refs, hyps)
    assert a == b
    assert a["wer_ci_low"] <= a["wer"] <= a["wer_ci_high"]
    assert a["wer_ci_low"] < a["wer_ci_high"]


def test_normalization_applied_to_both_sides(tmp_path):
    # Reference with Arabic kaf/yeh, diacritics and ۔; prediction in plain Urdu letters -> 0 errors.
    split = split_file(tmp_path, ["كيا آپ پَاکِسْتَان گئے۔"])
    pred = pred_file(tmp_path, ["کیا آپ پاکستان گئے"])
    row = ev.main(["--split", str(split), "--pred", str(pred), "--results", str(tmp_path / "r.csv")])
    assert row["wer"] == 0 and row["cer"] == 0


def test_digits_in_prediction_match_spelled_out_reference(tmp_path):
    # Real Kathbath reference; a model that writes "2015" heard the number correctly.
    split = split_file(tmp_path, ["دو ہزار پندرہ میں ایپل واچ کو لانچ کیا گیا"])
    pred = pred_file(tmp_path, ["2015 میں ایپل واچ کو لانچ کیا گیا"])
    row = ev.main(["--split", str(split), "--pred", str(pred), "--results", str(tmp_path / "r.csv")])
    assert row["wer"] == 0 and row["cer"] == 0


def test_three_normalization_levels(tmp_path):
    # Reference as in Kathbath; prediction like Whisper: madda, digits, punctuation.
    split = split_file(tmp_path, ["دو ہزار پندرہ میں اٹھ لوگ ائے"])
    pred = pred_file(tmp_path, ["2015 میں آٹھ لوگ آئے۔"])
    results = tmp_path / "r.csv"
    base = ["--split", str(split), "--pred", str(pred), "--results", str(results)]
    raw = ev.main(base + ["--no-normalize"])
    nodig = ev.main(base + ["--no-digits"])
    full = ev.main(base)
    assert (raw["split"], nodig["split"], full["split"]) == ("test_nonorm", "test_nodigits", "test")
    # raw: "2015"≠3 words, آٹھ≠اٹھ, "آئے۔"≠"ائے" -> 1 sub + 2 del + 2 sub = 5 errors / 7 words
    assert raw["wer"] == pytest.approx(5 / 7, abs=1e-6)
    # letters + punctuation fixed, digits not: "2015" vs "دو ہزار پندرہ" -> 1 sub + 2 del = 3 / 7
    assert nodig["wer"] == pytest.approx(3 / 7, abs=1e-6)
    assert full["wer"] == 0


def test_no_normalize_and_no_digits_cannot_be_combined(tmp_path):
    split = split_file(tmp_path, ["میں"])
    pred = pred_file(tmp_path, ["میں"])
    with pytest.raises(SystemExit):
        ev.main(["--split", str(split), "--pred", str(pred), "--no-normalize", "--no-digits",
                 "--results", str(tmp_path / "r.csv")])


def test_main_appends_row_with_expected_columns(tmp_path):
    split = split_file(tmp_path, ["میں گھر گیا", "وہ آیا"])
    pred = pred_file(tmp_path, ["میں گھر آیا", "وہ آیا"])
    results = tmp_path / "results.csv"
    ev.main(["--split", str(split), "--pred", str(pred), "--results", str(results)])
    ev.main(["--split", str(split), "--pred", str(pred), "--results", str(results), "--system", "other"])
    rows = list(csv.DictReader(open(results, encoding="utf-8")))
    assert list(rows[0]) == ev.RESULT_COLUMNS
    assert [r["system"] for r in rows] == ["sys_a", "other"]
    assert rows[0]["split"] == "test" and rows[0]["n_utts"] == "2"
    assert float(rows[0]["wer"]) == pytest.approx(0.2)  # 1 error / 5 words


def test_missing_prediction_fails(tmp_path):
    split = split_file(tmp_path, ["میں گھر گیا", "وہ آیا"])
    pred = write(tmp_path / "p.csv", ["utt_id", "hyp"], [["u0", "میں گھر گیا"]])
    with pytest.raises(SystemExit, match="no prediction"):
        ev.main(["--split", str(split), "--pred", str(pred), "--results", str(tmp_path / "r.csv")])


def test_wrong_header_and_duplicates_fail(tmp_path):
    split = split_file(tmp_path, ["میں"])
    bad = write(tmp_path / "bad.csv", ["id", "text"], [["u0", "میں"]])
    with pytest.raises(SystemExit, match="header"):
        ev.load_pairs(split, bad)
    dup = write(tmp_path / "dup.csv", ["utt_id", "hyp"], [["u0", "میں"], ["u0", "میں"]])
    with pytest.raises(SystemExit, match="duplicate"):
        ev.load_pairs(split, dup)


def test_prediction_file_with_bom_is_read(tmp_path):
    split = split_file(tmp_path, ["میں"])
    pred = tmp_path / "bom.csv"
    pred.write_text("utt_id,hyp\nu0,میں\n", encoding="utf-8-sig")
    _, refs, hyps = ev.load_pairs(split, pred)
    assert refs == hyps == ["میں"]


def test_several_prediction_files_in_one_call(tmp_path):
    split = split_file(tmp_path, ["میں گھر گیا"])
    a = pred_file(tmp_path, ["میں گھر گیا"], "sys_a.csv")
    b = pred_file(tmp_path, ["میں گھر آیا"], "sys_b.csv")
    results = tmp_path / "r.csv"
    ev.main(["--split", str(split), "--pred", str(a), str(b), "--results", str(results)])
    rows = list(csv.DictReader(open(results, encoding="utf-8")))
    assert [(r["system"], float(r["wer"])) for r in rows] == [("sys_a", 0), ("sys_b", pytest.approx(1 / 3))]
    with pytest.raises(SystemExit, match="single"):
        ev.main(["--split", str(split), "--pred", str(a), str(b), "--system", "x", "--results", str(results)])


def test_exclude_gives_nonoverlap_split(tmp_path):
    split = split_file(tmp_path, ["میں گھر گیا", "وہ آیا"])
    pred = pred_file(tmp_path, ["غلط غلط غلط", "وہ آیا"])
    exclude = write(tmp_path / "test_overlap.csv", ["utt_id", "text"], [["u0", "میں گھر گیا"]])
    row = ev.main(["--split", str(split), "--pred", str(pred), "--exclude", str(exclude),
                   "--results", str(tmp_path / "r.csv")])
    assert row["split"] == "test_nonoverlap" and row["n_utts"] == 1 and row["wer"] == 0
