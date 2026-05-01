#!/usr/bin/env python3
"""
Shout — Missing training runs
================================

Two training runs are still missing from the experiment suite:

  1. JOINT (UPSAMPLED) — the single most valuable remaining training run.
     The current joint adapter used plain concatenation. Sei has ~1.3× the
     training volume of ncx, so ncx was effectively under-trained. Joint
     underperformed both monolinguals by 7.6 WER (sei) and 12.3 WER (ncx).
     Two hypotheses explain this:
        (A) Capacity interference — adapter can't serve two langs at rank 32
        (B) Data dilution — ncx got fewer effective updates
     Upsampling ncx to match sei in training volume discriminates between
     these. Required for defending the joint-vs-monolingual claim in the
     paper; without it, the interpretation is hypothesis-laden.

  2. THIRD SEED (optional) — seed=7 for sei and ncx monolingual.
     The existing seed replicates (42, 13) give n=2, which supports a mean
     but not a meaningful std. Third seed gives n=3, enough for error bars.
     Joint-seed replicates are deliberately skipped: the joint training
     path lives in this script, not in the ablation suite, so adding joint
     replicates is future work.

Usage:
  # Joint with upsampling only (fastest path to disambiguated claim):
  python run_missing_training.py --joint_upsampled

  # Both joint-upsampled and third seed monolinguals:
  python run_missing_training.py --joint_upsampled --third_seed

  # Third seed only (if joint-upsampled was already done):
  python run_missing_training.py --third_seed

Safety:
  Output paths are checked against the reference adapter paths and the
  existing experiments/ tree. The script refuses to write to:
    - sei_lora_adapter/, ncx_lora_adapter/      (paper's B2 references)
    - ncx_asr_model/ncx-sei-joint-asr/          (XLS-R joint, archived)
    - min_experiments/joint_adapter/            (the existing Whisper joint)
    - experiments/seed_replicates/[sei,ncx]_seed{42,13}/  (already run)
"""

import argparse
import inspect
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

SCRIPT_DIR = Path(__file__).parent.resolve()

# Reference & existing outputs that MUST NOT be written to
SEI_REF = SCRIPT_DIR / "sei_lora_adapter"
NCX_REF = SCRIPT_DIR / "ncx_lora_adapter"
XLS_JOINT = SCRIPT_DIR / "ncx_asr_model" / "ncx-sei-joint-asr"
MIN_JOINT = SCRIPT_DIR / "min_experiments" / "joint_adapter"
EXISTING_SEEDS = [
    SCRIPT_DIR / "experiments" / "seed_replicates" / f"{lang}_seed{s}"
    for lang in ("sei", "ncx") for s in (42, 13)
]

# New outputs go here
MISSING_DIR = SCRIPT_DIR / "missing_training"

# --------------------------------------------------------------------------- #
# Safety
# --------------------------------------------------------------------------- #

def assert_safe(p: Path, purpose: str):
    forbidden = [SEI_REF, NCX_REF, XLS_JOINT, MIN_JOINT] + EXISTING_SEEDS
    pr = p.resolve()
    for f in forbidden:
        try:
            fr = f.resolve()
        except Exception:
            continue
        if pr == fr:
            raise SystemExit(f"REFUSING: {p} is a reference/existing path ({purpose})")
        try:
            pr.relative_to(fr)
            raise SystemExit(f"REFUSING: {p} is inside reference/existing path {f} ({purpose})")
        except ValueError:
            pass


# --------------------------------------------------------------------------- #
# CV data loading (same paths the other scripts use)
# --------------------------------------------------------------------------- #

DATA_ROOT_TEMPLATES = [
    "/var/tmp/soobenve_audio/mdc_data/{lang}/extracted",
    "/part/01/Tmp/soobenve/nlp_project/mdc_data/{lang}/extracted",
    "/part/01/Tmp/soobenve/mdc_data/{lang}/extracted",
    str(Path.home() / "Downloads/nlp_project/mdc_data/{lang}/extracted"),
    "./mdc_data/{lang}/extracted",
]


def find_cv_root(lang: str, override: Optional[str] = None) -> Path:
    if override:
        p = Path(override.format(lang=lang) if "{lang}" in override else override)
        if (p / lang / "extracted").exists():
            p = p / lang / "extracted"
        candidates = [p]
    else:
        candidates = [Path(t.format(lang=lang)) for t in DATA_ROOT_TEMPLATES]
    for root in candidates:
        if root.exists():
            for clips in root.rglob("clips"):
                parent = clips.parent
                if (parent / "test.tsv").exists():
                    return parent
    raise FileNotFoundError(
        f"Cannot find Common Voice data for '{lang}'. Tried: {[str(c) for c in candidates]}"
    )


# --------------------------------------------------------------------------- #
# Shared training helpers (same design as frozen whisper_lora_train.py)
# --------------------------------------------------------------------------- #

def _load_split(cv_root: Path, split: str):
    # NOTE: deliberately does NOT use datasets.Audio with cast_column,
    # because that uses torchcodec/FFmpeg under the hood and was the source
    # of the production bug in the earlier run (fixed by patching
    # prepare_dataset to use librosa.load directly on path strings).
    # We keep the "audio" column as a plain path string and let _prepare
    # call librosa.load at encoding time.
    import csv as _csv
    import pandas as pd
    from datasets import Dataset

    tsv_map = {"train": ["train.tsv", "validated.tsv"],
               "validation": ["dev.tsv"], "test": ["test.tsv"]}
    clips_dir = cv_root / "clips"
    for tsv_name in tsv_map[split]:
        tsv = cv_root / tsv_name
        if not tsv.exists():
            continue
        df = pd.read_csv(tsv, sep="\t", quoting=_csv.QUOTE_NONE,
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
                pe = cv_root / excl
                if pe.exists():
                    xdf = pd.read_csv(pe, sep="\t", quoting=_csv.QUOTE_NONE,
                                      keep_default_na=False, na_filter=False)
                    held_out |= set(xdf["path"].astype(str))
            orig_paths = df["audio_path"].map(lambda ap: Path(ap).name)
            df = df[~orig_paths.isin(held_out)].reset_index(drop=True)
        if len(df) == 0:
            continue
        ds = Dataset.from_dict({
            "audio": df["audio_path"].tolist(),      # plain path strings
            "sentence": df["sentence"].tolist(),
        })
        # NO cast_column(..., HFAudio(...)) here — see note at top of function
        return ds
    return None


def _upsample_to_match(datasets_by_lang: dict, seed: int):
    """Repeat smaller datasets until all reach the size of the largest.

    Uses Dataset.select(indices) with a deterministic repeat-then-truncate
    index list. The result has shape (target_size,) for each language —
    training sees equal per-epoch volume across languages.

    For a sei/ncx ratio of ~1.3:1, ncx entries get repeated ~1.3× on
    average. Larger ratios would repeat more aggressively; that's fine —
    the repetition is about gradient-update volume, not about introducing
    new information.
    """
    import random as _random
    from datasets import concatenate_datasets

    sizes = {lang: len(ds) for lang, ds in datasets_by_lang.items()}
    target = max(sizes.values())
    print(f"[upsample] target size per language: {target}")
    print(f"[upsample] before: {sizes}")

    rng = _random.Random(seed)
    out = []
    for lang, ds in datasets_by_lang.items():
        n = len(ds)
        if n == target:
            out.append(ds)
            continue
        # Repeat base indices enough times, then shuffle, then truncate
        n_repeats = (target + n - 1) // n  # ceil(target/n)
        indices = list(range(n)) * n_repeats
        rng.shuffle(indices)
        indices = indices[:target]
        upsampled = ds.select(indices)
        print(f"[upsample]   {lang}: {n} → {len(upsampled)} (repeat factor ~{target/n:.2f})")
        out.append(upsampled)
    return out


def _run_training(
    train_ds,
    eval_ds,
    output_dir: Path,
    seed: int,
    per_device_batch: int,
    grad_accum: int,
    num_train_epochs: int,
    learning_rate: float,
    wandb_run_name: str,
    no_wandb: bool,
    run_description: str,
):
    """Single training call. Identical hyperparameters to whisper_lora_train.py.

    Shared by both the joint-upsampled run and the third-seed monolingual runs
    so the recipe stays consistent.
    """
    import evaluate
    import numpy as np
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    # PEFT + Whisper compatibility patch
    _orig_fwd = WhisperForConditionalGeneration.forward
    _valid = set(inspect.signature(_orig_fwd).parameters.keys())

    def _patched_fwd(self, *args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in _valid}
        return _orig_fwd(self, *args, **kwargs)
    WhisperForConditionalGeneration.forward = _patched_fwd

    # Seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    processor = WhisperProcessor.from_pretrained("openai/whisper-small", task="transcribe")
    processor.tokenizer.set_prefix_tokens(language=None, task="transcribe")

    def _prepare(batch):
        # Load audio from path with librosa rather than relying on
        # datasets.Audio (which uses torchcodec/FFmpeg and breaks on the
        # production environment). Path string comes in via the "audio"
        # column; see _load_split.
        import librosa
        audio_path = batch["audio"]
        audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
        batch["input_features"] = processor.feature_extractor(
            audio_array, sampling_rate=sr
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    print(f"[train] Encoding features...")
    enc_train = train_ds.map(_prepare, remove_columns=train_ds.column_names,
                             num_proc=4, desc="encoding train")
    enc_eval = None
    if eval_ds is not None:
        enc_eval = eval_ds.map(_prepare, remove_columns=eval_ds.column_names,
                               num_proc=4, desc="encoding eval")

    print(f"[train] Loading whisper-small + LoRA(r=32)...")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=32, lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=0.05, bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    @dataclass
    class _Collator:
        processor: Any

        def __call__(self, features):
            input_feats = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(input_feats, return_tensors="pt")
            label_feats = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_feats, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch["attention_mask"].ne(1), -100
            )
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    wer_m = evaluate.load("wer", experiment_id=f"train-{os.getpid()}-wer")
    cer_m = evaluate.load("cer", experiment_id=f"train-{os.getpid()}-cer")

    def _metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pstr = processor.batch_decode(pred_ids, skip_special_tokens=True)
        lstr = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {
            "wer": wer_m.compute(predictions=pstr, references=lstr),
            "cer": cer_m.compute(predictions=pstr, references=lstr),
        }

    class _Trainer(Seq2SeqTrainer):
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            labels = inputs.pop("labels", None)
            loss, preds, _ = super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            if labels is not None:
                inputs["labels"] = labels
            return loss, preds, labels

    if no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ.setdefault("WANDB_PROJECT", "shout-missing")

    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_steps=50,
        num_train_epochs=num_train_epochs,
        gradient_checkpointing=False,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch" if enc_eval is not None else "no",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=128,
        load_best_model_at_end=enc_eval is not None,
        metric_for_best_model="wer" if enc_eval is not None else None,
        greater_is_better=False,
        save_total_limit=2,
        report_to="none" if no_wandb else "wandb",
        run_name=wandb_run_name,
        dataloader_num_workers=8,
        remove_unused_columns=False,
        seed=seed,
    )

    trainer = _Trainer(
        model=model, args=training_args,
        train_dataset=enc_train, eval_dataset=enc_eval,
        data_collator=_Collator(processor=processor),
        compute_metrics=_metrics,
        processing_class=processor.feature_extractor,
    )

    print(f"[train] Starting: {run_description}")
    trainer.train()

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    config = {
        "description": run_description,
        "base_model": "openai/whisper-small",
        "lora": {"r": 32, "alpha": 64, "dropout": 0.05,
                 "targets": ["q_proj", "v_proj", "k_proj", "out_proj"]},
        "per_device_batch": per_device_batch,
        "grad_accum": grad_accum,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "seed": seed,
    }
    with open(output_dir / "train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"[train] Adapter saved → {adapter_dir}")
    return adapter_dir


# --------------------------------------------------------------------------- #
# Joint training with upsampling
# --------------------------------------------------------------------------- #

def train_joint_upsampled(args):
    from datasets import concatenate_datasets

    output_dir = MISSING_DIR / "joint_upsampled"
    assert_safe(output_dir, "joint upsampled training")

    # Skip if already done
    adapter_path = output_dir / "adapter"
    if adapter_path.exists() and (output_dir / "train_config.json").exists():
        print(f"[joint-upsampled] SKIP — adapter already at {adapter_path}")
        return adapter_path

    print(f"\n{'='*65}\n  JOINT (UPSAMPLED) — disambiguates interference vs dilution\n{'='*65}\n")

    # Load both languages' train and validation splits
    lang_train = {}
    lang_val = []
    for lang in ("sei", "ncx"):
        cv_root = find_cv_root(lang, args.data_dir)
        print(f"  {lang} CV root: {cv_root}")
        train = _load_split(cv_root, "train")
        val = _load_split(cv_root, "validation")
        if train is None:
            raise RuntimeError(f"No training data for {lang} under {cv_root}")
        lang_train[lang] = train
        if val is not None:
            lang_val.append(val)

    # Upsample train so each language has equal volume
    upsampled = _upsample_to_match(lang_train, seed=args.seed)
    train_ds = concatenate_datasets(upsampled).shuffle(seed=args.seed)
    print(f"[joint-upsampled] combined train size: {len(train_ds)}")

    # Validation is NOT upsampled — it's for model selection on natural-distribution data
    eval_ds = concatenate_datasets(lang_val).shuffle(seed=args.seed) if lang_val else None
    if eval_ds is not None:
        print(f"[joint-upsampled] combined validation size: {len(eval_ds)}")

    _run_training(
        train_ds=train_ds, eval_ds=eval_ds,
        output_dir=output_dir,
        seed=args.seed,
        per_device_batch=args.per_device_batch,
        grad_accum=args.grad_accum,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        wandb_run_name="joint-upsampled",
        no_wandb=args.no_wandb,
        run_description="Joint sei+ncx with ncx upsampled to match sei volume",
    )
    return adapter_path


# --------------------------------------------------------------------------- #
# Third-seed monolingual replicates
# --------------------------------------------------------------------------- #

def train_third_seed(args, lang: str, seed: int):
    output_dir = MISSING_DIR / f"{lang}_seed{seed}"
    assert_safe(output_dir, f"{lang} seed={seed} training")

    adapter_path = output_dir / "adapter"
    if adapter_path.exists() and (output_dir / "train_config.json").exists():
        print(f"[{lang}-seed{seed}] SKIP — adapter already at {adapter_path}")
        return adapter_path

    print(f"\n{'='*65}\n  MONOLINGUAL {lang.upper()} SEED={seed} — third replicate for error bars\n{'='*65}\n")

    cv_root = find_cv_root(lang, args.data_dir)
    train = _load_split(cv_root, "train")
    val = _load_split(cv_root, "validation")
    if train is None:
        raise RuntimeError(f"No training data for {lang} under {cv_root}")
    print(f"  {lang} train: {len(train)}, val: {len(val) if val is not None else 0}")

    _run_training(
        train_ds=train, eval_ds=val,
        output_dir=output_dir,
        seed=seed,
        per_device_batch=args.per_device_batch,
        grad_accum=args.grad_accum,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        wandb_run_name=f"{lang}-seed{seed}",
        no_wandb=args.no_wandb,
        run_description=f"{lang} monolingual, seed={seed}, third replicate",
    )
    return adapter_path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Shout — missing training runs")
    parser.add_argument("--joint_upsampled", action="store_true",
                        help="Train joint adapter with language-balanced upsampling")
    parser.add_argument("--third_seed", action="store_true",
                        help="Train seed=7 for sei and ncx monolingual (third replicate)")
    parser.add_argument("--third_seed_value", type=int, default=7,
                        help="Seed value for third replicate (default: 7)")
    parser.add_argument("--data_dir", default=None,
                        help="Override CV data root")

    # Hyperparameters — match the existing joint/monolingual runs by default
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for joint-upsampled training")
    parser.add_argument("--per_device_batch", type=int, default=8,
                        help="A5000 24GB: 8 for whisper-small")
    parser.add_argument("--grad_accum", type=int, default=2,
                        help="Effective batch = per_device × grad_accum")
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()

    if not (args.joint_upsampled or args.third_seed):
        parser.error("Pass at least one of --joint_upsampled or --third_seed")

    MISSING_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'='*65}")
    print(f"  Shout — missing training runs")
    print(f"  Output root: {MISSING_DIR}")
    print(f"  Joint upsampled: {args.joint_upsampled}")
    print(f"  Third seed (seed={args.third_seed_value}): {args.third_seed}")
    print(f"{'='*65}")

    if args.joint_upsampled:
        train_joint_upsampled(args)

    if args.third_seed:
        for lang in ("sei", "ncx"):
            train_third_seed(args, lang, args.third_seed_value)

    print("\n✓ Done.")


if __name__ == "__main__":
    main()