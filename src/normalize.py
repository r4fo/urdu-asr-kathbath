"""Text normalization for Urdu ASR scoring.

evaluate.py applies normalize() to references and predictions in exactly the same way.
Do not normalize predictions yourself before writing a prediction file.

Rules, in order:
0. Numbers written with digits become Urdu words (src/urdu_numbers.py), e.g. "10,000" -> "دس ہزار",
   "2015" -> "دو ہزار پندرہ". This runs first, before punctuation is removed, so "10,000" stays one number.
   The references have no digits, so in practice this only changes predictions.
1. Unicode NFC, and ی + hamza (U+06CC U+0654) -> ئ (U+0626), which NFC does not compose.
2. Remove invisible format characters (Unicode category Cf: ZWNJ, ZWJ, BOM, RLM/LRM, direction
   embeddings), zero-width space (U+200B), and tatweel (U+0640).
3. Unify letter variants to the Urdu form:
     ك (U+0643 Arabic kaf)        -> ک (U+06A9 keheh)
     ي (U+064A Arabic yeh)        -> ی (U+06CC Farsi yeh)
     ى (U+0649 alef maksura)      -> ی (U+06CC)
     ه (U+0647 Arabic heh)        -> ہ (U+06C1 heh goal)
     ۀ (U+06C0) / ۂ (U+06C2)      -> ہ (U+06C1)
     آ (U+0622) / أ (U+0623) / إ (U+0625) -> ا (U+0627)
     ؤ (U+0624)                   -> و (U+0648)
     ۓ (U+06D3)                   -> ے (U+06D2)
   The Kathbath references never use آ أ ؤ ۂ ۓ (they write ا ا و ہ ے), but Whisper does.
   ئ (U+0626) and ء (U+0621) are kept: the references use them.
   ے (U+06D2 yeh barree) and ھ (U+06BE do-chashmi heh) are different letters and are kept.
4. Remove diacritics: U+064B-U+0655 (harakat, madda, hamza marks) and U+0670 (superscript alef).
5. Punctuation (Unicode category P*, which includes ۔ ، ؟ ؛ « ») becomes a space.
6. Any digits left (none after rule 0) are mapped to ASCII 0-9.
7. Latin letters are lowercased; whitespace is collapsed to single spaces and stripped.
"""
import re
import unicodedata

try:
    from src.urdu_numbers import digits_to_words
except ImportError:  # when src/ itself is on the path
    from urdu_numbers import digits_to_words

_ZERO_WIDTH = "​‌‍﻿ـ"
_LETTERS = {
    "ك": "ک",  # ك -> ک
    "ي": "ی",  # ي -> ی
    "ى": "ی",  # ى -> ی
    "ه": "ہ",  # ه -> ہ
    "ۀ": "ہ",  # ۀ -> ہ
    "ۂ": "ہ",  # ۂ -> ہ
    "آ": "ا",  # آ -> ا
    "أ": "ا",  # أ -> ا
    "إ": "ا",  # إ -> ا
    "ؤ": "و",  # ؤ -> و
    "ۓ": "ے",  # ۓ -> ے
}
_DIACRITICS = [chr(c) for c in range(0x064B, 0x0656)] + ["ٰ"]  # includes madda U+0653
_DIGITS = {chr(0x06F0 + i): str(i) for i in range(10)} | {chr(0x0660 + i): str(i) for i in range(10)}

_TABLE = str.maketrans({**{c: None for c in _ZERO_WIDTH}, **_LETTERS,
                        **{c: None for c in _DIACRITICS}, **_DIGITS})
_SPACES = re.compile(r"\s+")


def normalize(text: str, digits: bool = True) -> str:
    """Apply all rules. digits=False skips rule 0 (only for the normalization comparison table)."""
    text = text or ""
    if digits:
        text = digits_to_words(text)
    text = unicodedata.normalize("NFC", text)
    # NFC composes Arabic yeh + hamza into ئ but not Farsi yeh (ی) + hamza, so do that one by hand.
    text = text.replace("یٔ", "ئ")
    text = text.translate(_TABLE)
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch
                   for ch in text if unicodedata.category(ch) != "Cf")
    return _SPACES.sub(" ", text.lower()).strip()
