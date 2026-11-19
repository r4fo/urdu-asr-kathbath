"""Fine-tune Whisper on our Urdu training split (our contribution).

Settings fit an 8 GB GPU (RTX 4060): fp16, gradient checkpointing, batch 8
with 2 accumulation steps (effective batch 16). Dev WER is computed during
training on a fixed subset of the dev split to choose the best checkpoint.
The test split is never used here.

Checkpoints go to checkpoints/ (ignored by git: weights are too large for
GitHub). The best checkpoint is saved to <output-dir>/best.

Pipeline test on 50 clips (no split needed):
  python models/whisper_finetune/train.py --train-manifest data/clean/metadata.csv \
      --dev-manifest data/clean/metadata.csv --train-limit 50 --dev-limit 20 \
      --max-steps 20 --eval-steps 10 --output-dir checkpoints/whisper_test

Full run (after the split is frozen):
  python models/whisper_finetune/train.py --epochs 4 --output-dir checkpoints/whisper_small_ft
Resume after an interruption: add --resume
"""

import argparse
import re
import unicodedata
from dataclasses import dataclass

import jiwer
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperForConditionalGeneration, WhisperProcessor, set_seed)

# Light normalization for choosing checkpoints only; final scores use src/evaluate.py.
_MAP = str.maketrans({"آ": "ا", "أ": "ا", "إ": "ا", "ؤ": "و",
                      "ۂ": "ہ", "ۓ": "ے", "ي": "ی", "ى": "ی",
                      "ك": "ک", "ه": "ہ"})


def light_normalize(text):
    text = unicodedata.normalize("NFC", str(text)).translate(_MAP)
    text = re.sub(r"[ً-ٰٕ]", "", text)
    text = re.sub(r"[^؀-ۿݐ-ݿ\s]|[،؛؟۔]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class SpeechDataset(Dataset):
    def __init__(self, manifest, processor, limit=None):
        df = pd.read_csv(manifest, encoding="utf-8")
        self.df = df.head(limit) if limit else df
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        audio, sr = sf.read(row["path"], dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        feats = self.processor.feature_extractor(audio, sampling_rate=sr).input_features[0]
        labels = self.processor.tokenizer(row["text"]).input_ids
        return {"input_features": feats, "labels": labels}


@dataclass
class Collator:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features):
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]  # the model adds the start token itself
        batch["labels"] = labels
        return batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/whisper-small")
    p.add_argument("--revision", default=None)
    p.add_argument("--train-manifest", default="data/splits/train.csv")
    p.add_argument("--dev-manifest", default="data/splits/dev.csv")
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--dev-limit", type=int, default=300, help="dev clips used for checkpoint choice")
    p.add_argument("--output-dir", default="checkpoints/whisper_small_ft")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--eval-steps", type=int, default=250)
    p.add_argument("--optim", default="adamw_torch",
                   help="use adamw_bnb_8bit to save memory (needs: pip install bitsandbytes)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    processor = WhisperProcessor.from_pretrained(args.model, revision=args.revision,
                                                 language="urdu", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model, revision=args.revision)
    model.generation_config.language = "urdu"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None  # language and task are set above instead
    model.config.use_cache = False  # required with gradient checkpointing

    train_ds = SpeechDataset(args.train_manifest, processor, args.train_limit)
    dev_ds = SpeechDataset(args.dev_manifest, processor, args.dev_limit)
    print(f"Train: {len(train_ds)} clips, dev subset: {len(dev_ds)} clips")

    def compute_metrics(pred):
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        hyps = processor.batch_decode(pred.predictions, skip_special_tokens=True)
        refs = processor.batch_decode(label_ids, skip_special_tokens=True)
        hyps = [light_normalize(h) for h in hyps]
        refs = [light_normalize(r) for r in refs]
        return {"wer": jiwer.wer(refs, hyps), "cer": jiwer.cer(refs, hyps)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        optim=args.optim,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=25,
        dataloader_num_workers=2,
        report_to="none",
        seed=args.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=dev_ds,
        data_collator=Collator(processor, model.config.decoder_start_token_id),
        compute_metrics=compute_metrics, processing_class=processor.feature_extractor)

    trainer.train(resume_from_checkpoint=True if args.resume else None)
    best = f"{args.output_dir}/best"
    trainer.save_model(best)
    processor.save_pretrained(best)
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint} "
          f"(dev WER {trainer.state.best_metric:.4f}), saved to {best}")


if __name__ == "__main__":
    main()
