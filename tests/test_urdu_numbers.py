"""Tests for src/urdu_numbers.py. Expected strings are copied from Kathbath references."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.urdu_numbers import digits_to_words, number_to_words  # noqa: E402


def test_small_numbers():
    assert number_to_words(0) == "صفر"
    assert number_to_words(8) == "اٹھ"          # no madda, as in the references
    assert number_to_words(36) == "چھتیس"
    assert number_to_words(99) == "ننانوے"


def test_hundreds_and_thousands_use_ek():
    assert number_to_words(100) == "ایک سو"
    assert number_to_words(1000) == "ایک ہزار"
    assert number_to_words(218) == "دو سو اٹھارہ"


def test_south_asian_units():
    assert number_to_words(100000) == "ایک لاکھ"
    assert number_to_words(2000000) == "بیس لاکھ"
    assert number_to_words(2180000000) == "دو ارب اٹھارہ کروڑ"


def test_years():
    assert number_to_words(2015) == "دو ہزار پندرہ"
    assert number_to_words(2002) == "دو ہزار دو"
    assert number_to_words(1971) == "انیس سو اکہتر"
    assert number_to_words(1900) == "انیس سو"


def test_digits_in_text():
    # Kathbath reference: "دو ہزار پندرہ میں ایپل واچ کو لانچ کیا گیا"
    assert digits_to_words("2015 میں ایپل واچ کو لانچ کیا گیا") == \
        "دو ہزار پندرہ میں ایپل واچ کو لانچ کیا گیا"
    assert digits_to_words("۲۰۱۵ میں") == "دو ہزار پندرہ میں"   # Urdu digits
    assert digits_to_words("10,000 شامی") == "دس ہزار شامی"     # thousands separator


def test_text_without_digits_is_unchanged():
    assert digits_to_words("ایک لاکھ تک کا زیور") == "ایک لاکھ تک کا زیور"
