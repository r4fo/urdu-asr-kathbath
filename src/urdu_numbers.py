"""Convert digits to Urdu number words, written the way the Kathbath references write them.

The references never contain digits: numbers are spelled out ("دو ہزار پندرہ").
Zero-shot models often output digits ("2015"), which would count as errors even
when the model heard the number correctly. evaluate.py applies digits_to_words()
to every system's output, so all systems are scored the same way.

Style taken from the references (checked on data/clean/metadata.csv):
  - "ایک سو", "ایک ہزار", "ایک لاکھ" (hundreds/thousands start with ایک)
  - South Asian units: سو 100, ہزار 1,000, لاکھ 100,000, کروڑ 10,000,000, ارب 1,000,000,000
  - years 1100-1999 read with سو: 1971 -> "انیس سو اکہتر"; 2015 -> "دو ہزار پندرہ"
  - no madda: آٹھ is written اٹھ (the references never use آ)
"""

import re

UNITS = (
    "صفر ایک دو تین چار پانچ چھ سات اٹھ نو "
    "دس گیارہ بارہ تیرہ چودہ پندرہ سولہ سترہ اٹھارہ انیس "
    "بیس اکیس بائیس تئیس چوبیس پچیس چھبیس ستائیس اٹھائیس انتیس "
    "تیس اکتیس بتیس تینتیس چونتیس پینتیس چھتیس سینتیس اڑتیس انتالیس "
    "چالیس اکتالیس بیالیس تینتالیس چوالیس پینتالیس چھیالیس سینتالیس اڑتالیس انچاس "
    "پچاس اکیاون باون ترپن چون پچپن چھپن ستاون اٹھاون انسٹھ "
    "ساٹھ اکسٹھ باسٹھ تریسٹھ چونسٹھ پینسٹھ چھیاسٹھ سڑسٹھ اڑسٹھ انہتر "
    "ستر اکہتر بہتر تہتر چوہتر پچہتر چھہتر ستتر اٹھہتر اناسی "
    "اسی اکیاسی بیاسی تراسی چوراسی پچاسی چھیاسی ستاسی اٹھاسی نواسی "
    "نوے اکیانوے بانوے ترانوے چورانوے پچانوے چھیانوے ستانوے اٹھانوے ننانوے"
).split()
assert len(UNITS) == 100

SCALES = [(10**9, "ارب"), (10**7, "کروڑ"), (10**5, "لاکھ"), (1000, "ہزار"), (100, "سو")]

# ASCII, Arabic-Indic and Extended Arabic-Indic (Urdu) digits -> ASCII
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_NUMBER = re.compile(r"\d+(?:[,،]\d{2,3})*")


def number_to_words(n):
    if n < 100:
        return UNITS[n]
    if 1100 <= n <= 1999:  # years like 1971 are read "انیس سو اکہتر"
        rest = n % 100
        return f"{UNITS[n // 100]} سو" + (f" {UNITS[rest]}" if rest else "")
    words = []
    for value, name in SCALES:
        if n >= value:
            words.append(f"{number_to_words(n // value)} {name}")
            n %= value
    if n:
        words.append(UNITS[n])
    return " ".join(words)


def digits_to_words(text):
    """Replace every number written with digits by Urdu words."""
    text = str(text).translate(_DIGITS)

    def repl(match):
        digits = re.sub(r"[,،]", "", match.group())
        return f" {number_to_words(int(digits))} "

    return re.sub(r"\s+", " ", _NUMBER.sub(repl, text)).strip()
