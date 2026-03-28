#!/usr/bin/env python3
"""
Shout Project — Baseline B2: Whisper + LoRA Fine-tuning
==========================================================
Fine-tunes openai/whisper-small with LoRA adapters for a single
low-resource language. Adapters are saved separately from the base
model so they can be loaded at runtime for browser (WASM) deployment.

Usage:
    python whisper_lora_train.py --language sei
    python whisper_lora_train.py --language ncx

Dependencies:
    pip install transformers datasets peft accelerate evaluate jiwer \
                librosa soundfile wandb bitsandbytes
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import evaluate
import numpy as np
import torch
from datasets import Audio as HFAudio
from datasets import Dataset, DatasetDict
import inspect
import peft
from peft import LoraConfig, PeftModelForSeq2SeqLM, TaskType, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

# ---------------------------------------------------------------------------
# PEFT + Whisper compatibility patch
# PeftModelForSeq2SeqLM.forward is designed for T5/BART and injects T5-style
# kwargs (input_ids, inputs_embeds, attention_mask...) that Whisper doesn't
# accept. We whitelist ONLY the kwargs Whisper's forward actually supports.
# ---------------------------------------------------------------------------
_orig_whisper_fwd = WhisperForConditionalGeneration.forward
_whisper_valid_kwargs = set(inspect.signature(_orig_whisper_fwd).parameters.keys())

def _patched_whisper_fwd(self, *args, **kwargs):
    # Keep only kwargs that Whisper's forward actually accepts
    kwargs = {k: v for k, v in kwargs.items() if k in _whisper_valid_kwargs}
    return _orig_whisper_fwd(self, *args, **kwargs)

WhisperForConditionalGeneration.forward = _patched_whisper_fwd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LANGUAGE_CONFIG = {
    "sei": {"name": "Seri", "whisper_lang_code": None},   # Not in Whisper, use None
    "ncx": {"name": "Central Puebla Nahuatl", "whisper_lang_code": None},
}

# Base model — whisper-small is the sweet spot for browser deployment (~240MB)
BASE_MODEL_ID = "openai/whisper-small"

# LoRA hyperparameters — from LoRA-Whisper paper (Song et al., 2024)
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

# Training hyperparams
SEED = 42
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-3
NUM_TRAIN_EPOCHS = 10
WARMUP_STEPS = 50
MAX_AUDIO_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Whisper + LoRA training for Shout")
    parser.add_argument("--language", required=True, choices=list(LANGUAGE_CONFIG.keys()),
                        help="Language code to train on (sei or ncx)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to language data dir (auto-detected if not set)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save adapter weights")
    parser.add_argument("--wandb_project", type=str, default="shout-whisper-lora")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers — reuses existing extracted Common Voice data
# ---------------------------------------------------------------------------

def find_existing_cv_data(language: str) -> Optional[Path]:
    """Try to find already-downloaded Common Voice data from the Wav2Vec2 run."""
    candidates = [
        Path(f"/var/tmp/soobenve_audio/mdc_data/{language}/extracted"),
        Path(f"/part/01/Tmp/soobenve/nlp_project/mdc_data/{language}/extracted"),
        Path(f"/part/01/Tmp/soobenve/mdc_data/{language}/extracted"),
    ]
    for c in candidates:
        if c.exists():
            # Find the cv-corpus root
            for clips_dir in c.rglob("clips"):
                parent = clips_dir.parent
                if (parent / "test.tsv").exists():
                    print(f"Found existing data at: {parent}")
                    return parent
    return None


def load_split_tsv(cv_root: Path, split: str, language: str) -> Optional[Dataset]:
    """Load a Common Voice TSV split as a HuggingFace Dataset."""
    import csv
    import pandas as pd

    tsv_map = {"train": ["train.tsv", "validated.tsv"], "validation": ["dev.tsv"], "test": ["test.tsv"]}
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

            # Exclude dev/test from train for official_plus_extra strategy
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
            ds = ds.cast_column("audio", HFAudio(sampling_rate=16000))
            print(f"  {split}: {len(ds)} examples")
            return ds
    return None


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def prepare_dataset(batch: Dict, processor: WhisperProcessor) -> Dict:
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch


# ---------------------------------------------------------------------------
# Data collator — handles padding for Seq2Seq
# ---------------------------------------------------------------------------

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
        # Remove decoder start token if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def make_compute_metrics(processor: WhisperProcessor):
    rank = int(os.environ.get("RANK", "0"))
    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "shout")
    exp_base = f"{run_id}-rank{rank}"

    wer_metric = evaluate.load(
        "wer",
        process_id=0,
        num_process=1,
        experiment_id=f"{exp_base}-wer",
    )
    cer_metric = evaluate.load(
        "cer",
        process_id=0,
        num_process=1,
        experiment_id=f"{exp_base}-cer",
    )

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


# ---------------------------------------------------------------------------
# Custom Trainer — strips `labels` from generate() calls during evaluation.
# Whisper's generate() rejects `labels` (training-only kwarg), but the
# Seq2SeqTrainer passes the entire inputs dict to generate().
# ---------------------------------------------------------------------------

class ShoutSeq2SeqTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Pop labels so generate() doesn't crash; we return them ourselves
        # so that compute_metrics can still compute WER/CER.
        labels = inputs.pop("labels", None)
        loss, preds, _ = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys=ignore_keys
        )
        if labels is not None:
            inputs["labels"] = labels  # restore for safety
        return loss, preds, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    lang = args.language
    lang_name = LANGUAGE_CONFIG[lang]["name"]

    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Paths
    work_dir = Path(args.output_dir) if args.output_dir else Path(f"./shout-{lang}-lora")
    work_dir.mkdir(parents=True, exist_ok=True)

    # W&B
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    run_name = f"{lang}-whisper-lora"

    print(f"\n{'='*60}")
    print(f"  Shout: Whisper + LoRA  |  Language: {lang_name} ({lang})")
    print(f"  Base model: {BASE_MODEL_ID}")
    print(f"  LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}")
    print(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    cv_root = find_existing_cv_data(lang)
    if cv_root is None:
        raise FileNotFoundError(
            f"Could not find Common Voice data for '{lang}'. "
            "Run the Wav2Vec2 notebook first to download it, or set --data_dir."
        )

    print("Loading splits...")
    raw_datasets = DatasetDict({
        split: ds
        for split in ["train", "validation", "test"]
        if (ds := load_split_tsv(cv_root, split, lang)) is not None
    })
    print(f"Dataset: {raw_datasets}\n")

    # -----------------------------------------------------------------------
    # 2. Processor
    # -----------------------------------------------------------------------
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID, task="transcribe")

    # Disable language token forcing — our languages are not in Whisper vocab
    # so we let the model decode freely
    processor.tokenizer.set_prefix_tokens(language=None, task="transcribe")

    # -----------------------------------------------------------------------
    # 3. Encode audio → features
    # -----------------------------------------------------------------------
    print("Preparing features...")
    encoded_datasets = raw_datasets.map(
        lambda b: prepare_dataset(b, processor),
        remove_columns=raw_datasets["train"].column_names,
        num_proc=4,
        desc="Encoding audio",
    )

    # -----------------------------------------------------------------------
    # 4. Load base model + LoRA
    # -----------------------------------------------------------------------
    print(f"Loading {BASE_MODEL_ID} + LoRA...")
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        # Whisper's attention projection layers
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    # Use enable_input_require_grads() instead of gradient_checkpointing — required for PEFT
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # -----------------------------------------------------------------------
    # 5. Training
    # -----------------------------------------------------------------------
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    compute_metrics = make_compute_metrics(processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(work_dir / "checkpoints"),
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        gradient_checkpointing=False,  # Disabled — PEFT uses enable_input_require_grads instead
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=128,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to="wandb",
        run_name=run_name,
        dataloader_num_workers=8,
        remove_unused_columns=False,  # Required — patched forward uses *args/**kwargs
        seed=SEED,
    )

    trainer = ShoutSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded_datasets["train"],
        eval_dataset=encoded_datasets.get("validation"),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    print("\nStarting training...")
    train_result = trainer.train()

    # -----------------------------------------------------------------------
    # 6. Save — adapter only (not base model)! This is the browser-deployable artifact.
    # -----------------------------------------------------------------------
    adapter_dir = work_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Save only the LoRA adapter weights (~50MB instead of 240MB)
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    print(f"\n✅ LoRA adapter saved to: {adapter_dir}")
    print(f"   Adapter size: {sum(f.stat().st_size for f in adapter_dir.rglob('*') if f.is_file()) / 1e6:.1f} MB")

    # -----------------------------------------------------------------------
    # 7. Evaluate on test set
    # -----------------------------------------------------------------------
    if "test" in encoded_datasets:
        print("\nEvaluating on test set...")
        test_metrics = trainer.evaluate(encoded_datasets["test"], metric_key_prefix="test")
        print(f"\nTest WER: {test_metrics.get('test_wer', 'N/A'):.4f}")
        print(f"Test CER: {test_metrics.get('test_cer', 'N/A'):.4f}")

        with open(work_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

    print(f"\n✅ All done! Check your W&B run: {run_name}")
    print(f"   To load for inference:")
    print(f"   >>> from peft import PeftModel")
    print(f"   >>> from transformers import WhisperForConditionalGeneration")
    print(f"   >>> base = WhisperForConditionalGeneration.from_pretrained('{BASE_MODEL_ID}')")
    print(f"   >>> model = PeftModel.from_pretrained(base, '{adapter_dir}')")


if __name__ == "__main__":
    main()
