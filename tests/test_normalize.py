"""Unit tests for src/normalize.py on real Urdu examples, one test per rule."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.normalize import normalize  # noqa: E402


def test_already_normal_text_unchanged():
    assert normalize("میں پاکستان سے ہوں") == "میں پاکستان سے ہوں"


def test_arabic_kaf_to_urdu_keheh():
    assert normalize("كتاب") == "کتاب" == normalize("کتاب")


def test_arabic_yeh_and_alef_maksura_to_farsi_yeh():
    assert normalize("ي") == "ی"
    assert normalize("كيا") == normalize("کیا") == "کیا"
    assert normalize("عيسى") == "عیسی"


def test_yeh_barree_is_kept_separate():
    # ے (yeh barree) is a different letter from ی; "سے" and "سی" must stay different.
    assert normalize("سے") == "سے"
    assert normalize("سے") != normalize("سی")


def test_arabic_heh_to_heh_goal():
    assert normalize("كه") == "کہ" == normalize("کہ")
    assert normalize("یه") == "یہ"


def test_do_chashmi_heh_is_kept():
    assert normalize("بھی") == "بھی"
    assert normalize("بھی") != normalize("بہی")


def test_heh_with_hamza_composed_or_not():
    # "کمرۂ": precomposed ۂ and ہ + hamza both become ہ.
    assert normalize("کمرۂ") == normalize("کمرۂ") == "کمرہ"


def test_diacritics_removed():
    assert normalize("پَاکِسْتَان") == "پاکستان"
    assert normalize("مُحَمَّد") == "محمد"
    assert normalize("اللّٰہ") == "اللہ"        # shadda + superscript alef
    assert normalize("کتابً") == "کتاب"         # tanween


def test_alef_variants_to_plain_alef():
    # The references never use آ أ إ (they write ا), but Whisper does.
    assert normalize("آپ") == "اپ"
    assert normalize("أحمد") == normalize("احمد") == "احمد"
    assert normalize("إسلام") == "اسلام"
    assert normalize("ا\u0653پ") == "اپ"   # alef + combining madda


def test_waw_and_yeh_barree_with_hamza_unified():
    assert normalize("مؤثر") == "موثر"
    assert normalize("و\u0654") == "و"       # waw + combining hamza
    assert normalize("ہوۓ") == "ہوے"
    assert normalize("ے\u0654") == "ے"


def test_yeh_with_hamza_and_lone_hamza_kept():
    # The references use ئ and ء, so they stay.
    assert normalize("گئے") == "گئے"
    assert normalize("گی\u0654ے") == "گئے"   # Farsi yeh + hamza -> ئ, same as the precomposed form
    assert normalize("ء") == "ء"


def test_urdu_punctuation_removed():
    assert normalize("وہ گیا۔") == "وہ گیا"
    assert normalize("کیا وہ جائیں گے؟") == "کیا وہ جائیں گے"
    assert normalize("ایک، دو؛ تین") == "ایک دو تین"
    assert normalize("«خبر»") == "خبر"


def test_punctuation_between_words_becomes_space():
    assert normalize("گیا۔رہا") == "گیا رہا"
    assert normalize("پہلا,دوسرا") == "پہلا دوسرا"


def test_digits_become_urdu_words_like_the_references():
    # References spell numbers out; Whisper often writes digits.
    assert normalize("2015 میں") == "دو ہزار پندرہ میں"
    assert normalize("۲۰۱۵ میں") == normalize("٢٠١٥ میں") == "دو ہزار پندرہ میں"
    assert normalize("سن 1993 میں") == "سن انیس سو ترانوے میں"
    assert normalize("8") == "اٹھ"   # same spelling as normalize("آٹھ")
    assert normalize("آٹھ") == "اٹھ"


def test_thousands_separator_kept_before_punctuation_removed():
    assert normalize("10,000 شامی") == "دس ہزار شامی"
    assert normalize("10،000 شامی") == "دس ہزار شامی"   # Urdu comma
    assert normalize("یہ 2015۔") == "یہ دو ہزار پندرہ"


def test_digits_option_off_keeps_digits():
    assert normalize("2015 میں", digits=False) == "2015 میں"
    assert normalize("۲۰۱۵ میں", digits=False) == "2015 میں"   # still one digit form
    assert normalize("آٹھ", digits=False) == "اٹھ"             # letter rules still apply


def test_reference_without_digits_unchanged_by_number_step():
    assert normalize("سن دو ہزار گیارہ تک") == "سن دو ہزار گیارہ تک"


def test_zero_width_and_tatweel_removed():
    assert normalize("پاک‌ستان") == "پاکستان"
    assert normalize("﻿سلام") == "سلام"
    assert normalize("سـلام") == "سلام"


def test_direction_marks_removed():
    # RLM and right-to-left embedding marks are invisible but would make "میں" a different word.
    assert normalize("\u200fمیں\u200f گھر") == "میں گھر"
    assert normalize("\u202bگھر\u202c") == "گھر"
    assert normalize("میں\u00a0گھر") == "میں گھر"   # no-break space


def test_spaces_collapsed_and_stripped():
    assert normalize("  میں \t پاکستان\n سے ہوں ") == "میں پاکستان سے ہوں"


def test_latin_lowercased_and_empty_input():
    assert normalize("Whisper ماڈل") == "whisper ماڈل"
    assert normalize("") == ""
    assert normalize("۔،؟") == ""
    assert normalize(None) == ""


def test_idempotent():
    s = "كيا  آپ نے «پَاکِسْتَان» ۱۹۴۷ میں دیکھا؟"
    assert normalize(normalize(s)) == normalize(s)
