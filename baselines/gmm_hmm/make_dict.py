"""Grapheme lexicon for the GMM-HMM baseline: every word is spelled with its letters as units.

Input:  a Kaldi data folder with `text` (the train split) and optionally others to report OOV rates.
Output: <dict-dir>/lexicon.txt, nonsilence_phones.txt, silence_phones.txt, optional_silence.txt,
        extra_questions.txt (the files utils/prepare_lang.sh expects).

The vocabulary is the words of the TRAIN text only. Words of dev and test that are not in it cannot
be recognized; their rate (OOV rate) is printed and goes into the README.

Usage:
  python baselines/gmm_hmm/make_dict.py --train-text work/data/train/text --dict-dir work/data/local/dict \
      --check work/data/dev/text work/data/test/text
"""
import argparse
import os


def words_of(text_path):
    """Yield the words of a Kaldi `text` file (first column is the utterance ID)."""
    with open(text_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            yield from parts[1:]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train-text", required=True)
    parser.add_argument("--dict-dir", required=True)
    parser.add_argument("--check", nargs="*", default=[], help="text files to report the OOV rate for")
    args = parser.parse_args()

    vocab = sorted(set(words_of(args.train_text)))
    letters = sorted({ch for w in vocab for ch in w})
    os.makedirs(args.dict_dir, exist_ok=True)

    def write(name, lines):
        with open(os.path.join(args.dict_dir, name), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    write("lexicon.txt", ["!SIL sil", "<unk> spn"] + [f"{w} {' '.join(w)}" for w in vocab])
    write("nonsilence_phones.txt", letters)
    write("silence_phones.txt", ["sil", "spn"])
    write("optional_silence.txt", ["sil"])
    open(os.path.join(args.dict_dir, "extra_questions.txt"), "w").close()
    print(f"{len(vocab)} words, {len(letters)} letters")

    known = set(vocab)
    for path in args.check:
        toks = list(words_of(path))
        oov = sum(1 for w in toks if w not in known)
        print(f"{path}: OOV rate {100 * oov / max(len(toks), 1):.2f}% ({oov} of {len(toks)} words)")


if __name__ == "__main__":
    main()
