"""Tests for the transcript and audio checks in scripts/clean.py."""

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "clean", Path(__file__).resolve().parents[1] / "scripts" / "clean.py")
clean = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clean)


def test_arabic_letter_variants_become_urdu():
    # Arabic yeh, kaf and heh -> Urdu choti ye, kaf and choti he
    text, fixes = clean.fix_text("كتاب يه")
    assert text == "کتاب یہ"
    assert len(fixes) == 3


def test_urdu_text_is_unchanged():
    urdu = "یہ کتاب ہے"
    text, fixes = clean.fix_text(urdu)
    assert text == urdu
    assert fixes == []


def test_digits_and_invisible_characters():
    text, fixes = clean.fix_text("١٢​ ـا")
    assert text == "۱۲ ا"
    assert any("U+200B" in f for f in fixes)


def test_whitespace_is_collapsed():
    text, _ = clean.fix_text("  ا   ب  ")
    assert text == "ا ب"


def test_chars_per_second_counts_letters_only():
    assert clean.chars_per_second("اب ت۔", 2.0) == 2.0


def test_silent_and_clipped_audio():
    silent_db, _ = clean.audio_checks(np.zeros(16000, dtype=np.float32))
    assert silent_db == float("-inf")
    tone = 0.1 * np.sin(np.linspace(0, 2000, 16000)).astype(np.float32)
    tone_db, tone_clip = clean.audio_checks(tone)
    assert -30 < tone_db < -15 and tone_clip == 0.0
    _, clipped = clean.audio_checks(np.ones(16000, dtype=np.float32))
    assert clipped == 1.0
