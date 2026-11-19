"""Fine-tune wav2vec 2.0 (XLS-R) with a CTC head on our Urdu training split.

This is our second contribution model: a CTC encoder, to compare with the
encoder-decoder Whisper. The CTC vocabulary is the set of characters in the
training transcripts (space becomes "|"), saved to <output-dir>/vocab.json.

Settings fit an 8 GB GPU: fp16, gradient checkpointing, frozen CNN feature
encoder, batch 8 with 2 accumulation steps, clips grouped by length to reduce
padding. Dev WER (greedy decoding, on a fixed dev subset) chooses the best
checkpoint. The test split is never used here.

Pipeline test on 50 clips (no split needed):
  python models/wav2vec2_finetune/train.py --train-manifest data/clean/metadata.csv \
      --dev-manifest data/clean/metadata.csv --train-limit 50 --dev-limit 20 \
      --max-steps 30 --eval-steps 15 --output-dir checkpoints/w2v2_test

Full run (after the split is frozen):
  python models/wav2vec2_finetune/train.py --epochs 10 --output-dir checkpoints/w2v2_xlsr_ft
Resume after an interruption: add --resume. Out of memory: --batch-size 4 --grad-accum 4
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import (Trainer, TrainingArguments, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,
                          Wav2Vec2ForCTC, Wav2Vec2Processor, set_seed)
from transformers.trainer_pt_utils import LengthGroupedSampler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.normalize import normalize  # noqa: E402  (only for the dev WER that chooses checkpoints)


def build_vocab(texts, path):
    chars = sorted(set("".join(texts)) - {" "})
    vocab = {c: i for i, c in enumerate(chars)}
    vocab["|"] = len(vocab)      # word separator (space)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)  # also the CTC blank
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab, ensure_ascii=False, indent=1), encoding="utf-8")
    return vocab


class SpeechDataset(Dataset):
    def __init__(self, manifest, processor, limit=None):
        df = pd.read_csv(manifest, encoding="utf-8")
        self.df = (df.head(limit) if limit else df).reset_index(drop=True)
        self.processor = processor
        self.lengths = self.df["duration"].tolist() if "duration" in self.df else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        audio, sr = sf.read(row["path"], dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        values = self.processor.feature_extractor(audio, sampling_rate=sr).input_values[0]
        labels = self.processor.tokenizer(" ".join(str(row["text"]).split())).input_ids
        return {"input_values": values, "labels": labels}


class LengthGroupedTrainer(Trainer):
    """Groups clips of similar duration into batches (less padding) using the manifest durations,
    so the audio does not have to be read once just to measure its length."""

    def _get_train_sampler(self, *args, **kwargs):
        lengths = self.train_dataset.lengths
        if not lengths:
            return super()._get_train_sampler(*args, **kwargs)
        return LengthGroupedSampler(self.args.train_batch_size * self.args.gradient_accumulation_steps,
                                    lengths=lengths)


@dataclass
class Collator:
    processor: Wav2Vec2Processor

    def __call__(self, features):
        batch = self.processor.feature_extractor.pad(
            [{"input_values": f["input_values"]} for f in features], padding=True, return_tensors="pt")
        labels = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], padding=True, return_tensors="pt")
        batch["labels"] = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
        return batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/wav2vec2-xls-r-300m")
    p.add_argument("--revision", default=None)
    p.add_argument("--train-manifest", default="data/splits/train.csv")
    p.add_argument("--dev-manifest", default="data/splits/dev.csv")
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--dev-limit", type=int, default=300)
    p.add_argument("--output-dir", default="checkpoints/w2v2_xlsr_ft")
    p.add_argument("--epochs", type=float, default=10.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    out = Path(args.output_dir)
    train_texts = pd.read_csv(args.train_manifest, encoding="utf-8")["text"]
    if args.train_limit:
        train_texts = train_texts.head(args.train_limit)
    vocab = build_vocab([" ".join(str(t).split()) for t in train_texts], out / "vocab.json")
    tokenizer = Wav2Vec2CTCTokenizer(str(out / "vocab.json"), unk_token="[UNK]",
                                     pad_token="[PAD]", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                 do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    print(f"Vocabulary: {len(vocab)} symbols")

    model = Wav2Vec2ForCTC.from_pretrained(
        args.model, revision=args.revision, ctc_loss_reduction="mean", ctc_zero_infinity=True,
        pad_token_id=tokenizer.pad_token_id, vocab_size=len(tokenizer),
        attention_dropout=0.0, hidden_dropout=0.0, feat_proj_dropout=0.0,
        layerdrop=0.05, mask_time_prob=0.05)
    model.freeze_feature_encoder()

    train_ds = SpeechDataset(args.train_manifest, processor, args.train_limit)
    dev_ds = SpeechDataset(args.dev_manifest, processor, args.dev_limit)
    print(f"Train: {len(train_ds)} clips, dev subset: {len(dev_ds)} clips")

    def compute_metrics(pred):
        ids = np.argmax(pred.predictions, axis=-1)
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        hyps = [normalize(h) for h in processor.batch_decode(ids)]
        refs = [normalize(r) for r in processor.batch_decode(label_ids, group_tokens=False)]
        return {"wer": jiwer.wer(refs, hyps), "cer": jiwer.cer(refs, hyps)}

    training_args = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_steps=50,
        dataloader_num_workers=2,
        report_to="none",
        seed=args.seed,
    )
    trainer = LengthGroupedTrainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=dev_ds,
                                   data_collator=Collator(processor), compute_metrics=compute_metrics,
                                   processing_class=processor.feature_extractor)
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    best = out / "best"
    trainer.save_model(str(best))
    processor.save_pretrained(str(best))
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint} "
          f"(dev WER {trainer.state.best_metric:.4f}), saved to {best}")


if __name__ == "__main__":
    main()
