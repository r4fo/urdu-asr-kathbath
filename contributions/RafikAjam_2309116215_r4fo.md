# Rafik Ajam

- Student ID: 2309116215
- GitHub: [r4fo](https://github.com/r4fo)
- Role: Member 1, data, our fine-tuned systems, repository and coordination

## What I did

- **Coordination and repository:** chose the language and dataset with the team, divided the work, set up the
  repository structure, `README.md`, `LICENSE`, `.gitignore` and the pinned `requirements.txt`.
- **Data:** `scripts/download_kathbath_urdu.py` (25-hour subset with a per-speaker hour cap, seed 42, 16-bit FLAC
  in `data/raw/`).
- **Cleaning and checks:** `scripts/clean.py` and `tests/test_clean.py` (11 automatic checks, cleaning log),
  `scripts/profile_data.py` (audio profiling), listening to the 10 most extreme clips (`data/clean/manual_review.csv`,
  1 removed) and `scripts/find_label_errors.py` (Whisper-based transcript check).
- **Whisper-small fine-tuning:** `models/whisper_finetune/train.py` and `transcribe.py`; trained on my RTX 4060,
  chose the checkpoint and beam on dev, ran test once. Weights on the Hub: `r4fo/whisper-small-urdu-kathbath`.
- **wav2vec 2.0 XLS-R 300M fine-tuning:** `models/wav2vec2_finetune/train.py` and `transcribe.py`; 15 epochs, chosen
  on dev. Weights on the Hub: `r4fo/wav2vec2-xls-r-300m-urdu-kathbath`.
- **KenLM 3-gram:** `build_lm.sh`, `tune_lm.py` and `kenlm_colab.ipynb`; LM from train text only, `alpha` and
  `beta` chosen on dev, one test run on Colab.
- **Pretrained baselines** (originally assigned to Member 5, not delivered by 22 November, so I ran them):
  Whisper-large-v3 zero-shot, anuragshas with and without its LM (`baselines/pretrained/anuragshas_lm_colab.ipynb`),
  kingabzpro.
- **Results and analysis:** ran `src/evaluate.py` for the systems above and the other members' prediction files,
  `scripts/make_tables.py`, `scripts/error_analysis.py`, and the README's results, error analysis and limitations.
- **Presentation:** the slides, presented by Mohamed Aziz Ben Thabet.

## How I used AI

I used Claude (Anthropic) to help plan the work, write and debug code, draft the README and the slides, and work
through setup errors (Windows, Colab, KenLM). I ran every training, decoding and evaluation command myself, on my
own machine or my own Colab session, and checked the outputs and numbers before committing them.
