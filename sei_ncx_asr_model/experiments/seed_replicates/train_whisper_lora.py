#!/usr/bin/env python3
"""
Shout — Flexible Whisper+LoRA training script for ablations
=============================================================

Parallel to the frozen `whisper_lora_train.py` which produced the paper's
original sei_lora_adapter/ and ncx_lora_adapter/ artefacts. That script
hard-codes rank=32, seed=42, all-attention-modules LoRA placement, and
full-dataset training. Changing any of those would risk overwriting the
reference artefacts.

This script takes the same data-loading and training loop but exposes:
  --lora_rank                 LoRA r (ablation Tier-1b)
  --lora_target               encoder-only / decoder-only / both (Tier-1c)
  --seed                      Reproducibility seed (Tier-1a seed replicates)
  --data_fraction             Subsample training data (Tier-2a learning curve)
  --base_model                whisper-{tiny,base,small,medium} (Tier-2b)
  --output_dir                Where the adapter + metrics go
  --wandb_run_name            W&B run name; different per experiment

Every run is isolated to its own output_dir. Running this script will NEVER
write to sei_lora_adapter/ or ncx_lora_adapter/; it writes under the path
the caller specifies.

Usage examples:
  # Seed replicate for ncx
  python train_whisper_lora.py --language ncx --seed 13 \\
    --output_dir experiments/seed_replicates/ncx_seed13

  # LoRA rank ablation
  python train_whisper_lora.py --language ncx --lora_rank 8 \\
    --output_dir experiments/rank_ablation/ncx_r8

  # Decoder-only LoRA
  python train_whisper_lora.py --language ncx --lora_target decoder \\
    --output_dir experiments/placement/ncx_decoder

  # 30-minute subsample
  python train_whisper_lora.py --language ncx --data_fraction 0.03 \\
    --output_dir experiments/data_curve/ncx_30min_seed42
"""

import argparse
import inspect
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import evaluate
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# --------------------------------------------------------------------------- #
# PEFT + Whisper compatibility patch (identical to the frozen script)         #
# --------------------------------------------------------------------------- #
_orig_whisper_fwd = WhisperForConditionalGeneration.forward
_whisper_valid_kwargs = set(inspect.signature(_orig_whisper_fwd).parameters.keys())


def _patched_whisper_fwd(self, *args, **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if k in _whisper_valid_kwargs}
    return _orig_whisper_fwd(self, *args, **kwargs)


WhisperForConditionalGeneration.forward = _patched_whisper_fwd

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #

LANGUAGE_CONFIG = {
    "sei": {"name": "Seri"},
    "ncx": {"name": "Central Puebla Nahuatl"},
}

BASE_MODEL_DEFAULTS = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v3",  # provided for completeness; not recommended for deployment
}

# LoRA target module presets.
# Whisper encoder and decoder both have q/k/v/out_proj in their attention blocks.
# The model module names are e.g. "model.encoder.layers.N.self_attn.q_proj".
# Using fully-qualified regex-like matches keeps placement clean.
LORA_TARGETS = {
    "both": ["q_proj", "v_proj", "k_proj", "out_proj"],
    # encoder-only: match names containing "encoder." followed by the attn projection
    "encoder": [
        "encoder.layers.*.self_attn.q_proj",
        "encoder.layers.*.self_attn.v_proj",
        "encoder.layers.*.self_attn.k_proj",
        "encoder.layers.*.self_attn.out_proj",
    ],
    # decoder-only: includes both self-attn AND cross-attn (encoder-decoder attention)
    "decoder": [
        "decoder.layers.*.self_attn.q_proj",
        "decoder.layers.*.self_attn.v_proj",
        "decoder.layers.*.self_attn.k_proj",
        "decoder.layers.*.self_attn.out_proj",
        "decoder.layers.*.encoder_attn.q_proj",
        "decoder.layers.*.encoder_attn.v_proj",
        "decoder.layers.*.encoder_attn.k_proj",
        "decoder.layers.*.encoder_attn.out_proj",
    ],
}

# PEFT accepts target_modules as a list of suffixes OR a regex pattern.
# We convert the wildcards to regex strings.
def _targets_to_regex(patterns: List[str]) -> str:
    import re
    parts = [p.replace(".", r"\.").replace("*", r"\d+") for p in patterns]
    return "(" + "|".join(parts) + ")$"


# --------------------------------------------------------------------------- #
# Argument parsing                                                             #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Flexible Whisper+LoRA for ablations")
    p.add_argument("--language", required=True, choices=list(LANGUAGE_CONFIG.keys()))
    p.add_argument("--output_dir", required=True, type=str,
                   help="Per-experiment output directory (must not overwrite reference adapters)")

    # LoRA knobs
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=None,
                   help="Defaults to 2*lora_rank if unset (matches Song et al. 2024)")
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target", choices=list(LORA_TARGETS.keys()), default="both")

    # Model size
    p.add_argument("--base_model", choices=list(BASE_MODEL_DEFAULTS.keys()),
                   default="small")

    # Training knobs
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per_device_batch", type=int, default=16,
                   help="Per-device batch size. A100 80GB handles 16 for whisper-small; "
                        "reduce to 8 for whisper-medium")
    p.add_argument("--grad_accum", type=int, default=1,
                   help="Set together with per_device_batch to maintain effective batch 64")
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--num_train_epochs", type=int, default=10)
    p.add_argument("--warmup_steps", type=int, default=50)

    # Data knobs
    p.add_argument("--data_fraction", type=float, default=1.0,
                   help="Fraction of training data to use (for learning-curve experiments). "
                        "1.0 = full, 0.5 = half, etc. Subsample is seed-deterministic.")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Override CV data root auto-detection")

    # Logging
    p.add_argument("--wandb_project", type=str, default="shout-ablations")
    p.add_argument("--wandb_run_name", type=str, default=None,
                   help="Defaults to {language}-r{rank}-{target}-seed{seed}")
    p.add_argument("--no_wandb", action="store_true", help="Disable W&B logging entirely")

    return p.parse_args()


# --------------------------------------------------------------------------- #
# Data loading (borrowed verbatim from whisper_lora_train.py)                  #
# --------------------------------------------------------------------------- #

def find_existing_cv_data(language: str, override: Optional[str] = None) -> Optional[Path]:
    if override:
        p = Path(override)
        if p.exists():
            return p
        raise FileNotFoundError(f"--data_dir {override} does not exist")
    candidates = [
        Path(f"/var/tmp/soobenve_audio/mdc_data/{language}/extracted"),
        Path(f"/part/01/Tmp/soobenve/nlp_project/mdc_data/{language}/extracted"),
        Path(f"/part/01/Tmp/soobenve/mdc_data/{language}/extracted"),
        Path(f"./mdc_data/{language}/extracted"),
    ]
    for c in candidates:
        if c.exists():
            for clips_dir in c.rglob("clips"):
                parent = clips_dir.parent
                if (parent / "test.tsv").exists():
                    return parent
    return None


def load_split_tsv(cv_root: Path, split: str, language: str) -> Optional[Dataset]:
    import csv
    import pandas as pd

    tsv_map = {"train": ["train.tsv", "validated.tsv"],
               "validation": ["dev.tsv"], "test": ["test.tsv"]}
    clips_dir = cv_root / "clips"

    for tsv_name in tsv_map.get(split, []):
        tsv_path = cv_root / tsv_name
        if tsv_path.exists():
            df = pd.read_csv(tsv_path, sep="\t", quoting=csv.QUOTE_NONE,
                             keep_default_na=False, na_filter=False)
            text_col = "sentence" if "sentence" in df.columns else "text"
            df["audio_path"] = df["path"].astype(str).map(
                lambda p: str((clips_dir / p).resolve()))
            df["sentence"] = df[text_col].astype(str).str.lower()
            df = df[df["audio_path"].map(lambda p: Path(p).exists())]
            df = df[["audio_path", "sentence"]].dropna().reset_index(drop=True)

            if split == "train" and tsv_name == "validated.tsv":
                held_out = set()
                for excl in ["dev.tsv", "test.tsv"]:
                    p = cv_root / excl
                    if p.exists():
                        excl_df = pd.read_csv(p, sep="\t", quoting=csv.QUOTE_NONE,
                                              keep_default_na=False, na_filter=False)
                        held_out |= set(excl_df["path"].astype(str))
                original_paths = df["audio_path"].map(lambda ap: Path(ap).name)
                df = df[~original_paths.isin(held_out)].reset_index(drop=True)

            if len(df) == 0:
                continue

            ds = Dataset.from_dict({
                "audio": df["audio_path"].tolist(),
                "sentence": df["sentence"].tolist(),
            })
            return ds
    return None


def subsample_dataset(ds: Dataset, fraction: float, seed: int) -> Dataset:
    """Deterministic subsample for learning-curve experiments.
    Shuffled first so we don't systematically pick early recordings."""
    if fraction >= 1.0:
        return ds
    shuffled = ds.shuffle(seed=seed)
    n = max(1, int(len(shuffled) * fraction))
    return shuffled.select(range(n))


# --------------------------------------------------------------------------- #
# Feature extraction / collator / metrics                                      #
# --------------------------------------------------------------------------- #

def prepare_dataset(batch: Dict, processor: WhisperProcessor) -> Dict:
    import librosa
    audio_path = batch["audio"]
    audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
    batch["input_features"] = processor.feature_extractor(
        audio_array, sampling_rate=sr
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def make_compute_metrics(processor: WhisperProcessor):
    rank_id = int(os.environ.get("RANK", "0"))
    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "shout")
    exp_base = f"{run_id}-rank{rank_id}-{os.getpid()}"

    wer_metric = evaluate.load("wer", experiment_id=f"{exp_base}-wer")
    cer_metric = evaluate.load("cer", experiment_id=f"{exp_base}-cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {
            "wer": wer_metric.compute(predictions=pred_str, references=label_str),
            "cer": cer_metric.compute(predictions=pred_str, references=label_str),
        }

    return compute_metrics


class ShoutSeq2SeqTrainer(Seq2SeqTrainer):
    """Strips `labels` from generate() calls during evaluation (identical to frozen)."""

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels = inputs.pop("labels", None)
        loss, preds, _ = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys=ignore_keys
        )
        if labels is not None:
            inputs["labels"] = labels
        return loss, preds, labels


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    lang = args.language
    lang_name = LANGUAGE_CONFIG[lang]["name"]

    # Output-dir safety check — never overwrite reference adapters
    output_dir = Path(args.output_dir)
    forbidden = [
        Path("./sei_lora_adapter"),
        Path("./ncx_lora_adapter"),
        Path("./ncx_asr_model/ncx-sei-joint-asr"),
    ]
    for f in forbidden:
        if output_dir.resolve() == f.resolve():
            raise SystemExit(
                f"Refusing to write to reference artefact path {f}. "
                "Pick a different --output_dir."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Resolve model + lora config
    base_model_id = BASE_MODEL_DEFAULTS[args.base_model]
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_rank * 2

    run_name = args.wandb_run_name or (
        f"{lang}-{args.base_model}-r{args.lora_rank}-{args.lora_target}-seed{args.seed}"
        + (f"-frac{args.data_fraction:.2f}" if args.data_fraction < 1.0 else "")
    )

    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    print(f"\n{'='*65}")
    print(f"  Shout ablation — {lang_name} ({lang})")
    print(f"  Output: {output_dir}")
    print(f"  Base model: {base_model_id}")
    print(f"  LoRA rank={args.lora_rank} alpha={lora_alpha} target={args.lora_target}")
    print(f"  Seed={args.seed}  Data fraction={args.data_fraction}")
    print(f"  Effective batch = {args.per_device_batch * args.grad_accum * max(1, torch.cuda.device_count())}")
    print(f"{'='*65}\n")

    # Save config so every run is self-describing
    config_dump = {
        **vars(args),
        "base_model_id": base_model_id,
        "lora_alpha": lora_alpha,
        "run_name": run_name,
    }
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(config_dump, f, indent=2)

    # 1. Load data
    cv_root = find_existing_cv_data(lang, args.data_dir)
    if cv_root is None:
        raise FileNotFoundError(f"Could not find Common Voice data for '{lang}'.")
    print(f"CV root: {cv_root}")

    raw_datasets = DatasetDict({
        split: ds
        for split in ["train", "validation", "test"]
        if (ds := load_split_tsv(cv_root, split, lang)) is not None
    })
    for split, ds in raw_datasets.items():
        print(f"  {split}: {len(ds)}")

    if args.data_fraction < 1.0 and "train" in raw_datasets:
        before = len(raw_datasets["train"])
        raw_datasets["train"] = subsample_dataset(
            raw_datasets["train"], args.data_fraction, args.seed
        )
        print(f"  [subsampled train: {before} -> {len(raw_datasets['train'])} "
              f"at fraction {args.data_fraction}]")

    # 2. Processor
    processor = WhisperProcessor.from_pretrained(base_model_id, task="transcribe")
    processor.tokenizer.set_prefix_tokens(language=None, task="transcribe")

    # 3. Encode
    encoded = raw_datasets.map(
        lambda b: prepare_dataset(b, processor),
        remove_columns=raw_datasets["train"].column_names,
        num_proc=4,
        desc="Encoding audio",
    )

    # 4. Build model + LoRA
    print(f"Loading {base_model_id} + LoRA...")
    model = WhisperForConditionalGeneration.from_pretrained(base_model_id)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    target_patterns = LORA_TARGETS[args.lora_target]
    if args.lora_target == "both":
        # Simple suffix match works for both-mode
        target_modules = target_patterns
    else:
        target_modules = _targets_to_regex(target_patterns)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # 5. Training
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    compute_metrics = make_compute_metrics(processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        gradient_checkpointing=False,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=128,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none" if args.no_wandb else "wandb",
        run_name=run_name,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = ShoutSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded.get("validation"),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    print("\nStarting training...")
    train_result = trainer.train()

    # 6. Save adapter
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    # 7. Evaluate on test set and save per-clip predictions (matches eval_b1 schema)
    if "test" in encoded:
        print("\nEvaluating on test set...")
        test_metrics = trainer.evaluate(encoded["test"], metric_key_prefix="test")

        # Use the same batch_decode to get per-clip predictions
        test_pred = trainer.predict(encoded["test"])
        pred_ids = test_pred.predictions
        label_ids = test_pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        import csv
        pred_csv = output_dir / "test_predictions.csv"
        with open(pred_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["language", "reference", "prediction"])
            w.writeheader()
            for ref, pred in zip(label_str, pred_str):
                w.writerow({"language": lang, "reference": ref, "prediction": pred})
        print(f"  Predictions → {pred_csv}")

        with open(output_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"  Test WER: {test_metrics.get('test_wer', float('nan')):.4f}")
        print(f"  Test CER: {test_metrics.get('test_cer', float('nan')):.4f}")

    print(f"\n✓ Done. Artefacts in {output_dir}")


if __name__ == "__main__":
    main()
