#!/usr/bin/env python3
"""
Shout — Minimum experiment suite (single-file runner)
========================================================

Runs the complete minimum experiment plan in one invocation:

  1. Fill missing B2-ncx test metrics (existing ncx_lora_adapter has no test_metrics.json)
  2. Train the joint Whisper+LoRA adapter on concatenated sei+ncx data
  3. Evaluate every adapter × test-set pair:
        sei adapter  → sei test
        ncx adapter  → ncx test
        joint adapter → sei test
        joint adapter → ncx test
        sei adapter  → ncx test   (off-diagonal — controls for interference)
        ncx adapter  → sei test   (off-diagonal — controls for interference)
     Each evaluation produces test_predictions.csv + test_tokens.jsonl.
  4. For every prediction set, run the τ-sweep over {0.0, 0.3, 0.5, 0.7, 1.0}
     which covers B4 (τ=1.0, always-on) and B4.5 (τ<1.0, uncertainty-triggered)
     in one pass. τ=0.0 equals passthrough (no reconstruction).
  5. For each τ=1.0 sweep result, compute morpheme F1 via eval_morpheme_f1.py
     conventions (writes b4_*_reconstruction.csv files that eval_morpheme_f1
     accepts directly).
  6. Emit one consolidated results table.

Properties:
  - Idempotent. Re-running skips anything already complete.
  - Safe. Will not write to sei_lora_adapter/, ncx_lora_adapter/, or
    ncx_asr_model/ncx-sei-joint-asr/ (paper's reference artefacts).
  - Self-contained. Does not require the earlier ablation suite to be present.
  - Single A100 sufficient. Total compute ~2 hours.

Usage:
  cd /path/to/sei_ncx_asr_model
  python run_minimum_experiments.py
  python run_minimum_experiments.py --skip-train          # reuse existing joint if present
  python run_minimum_experiments.py --only eval           # skip training, run evals only
  python run_minimum_experiments.py --only tau_sweep      # just the sweeps (needs evals done)
  python run_minimum_experiments.py --data_dir /path      # override CV data location
"""

import argparse
import csv
import inspect
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Paths & output layout                                                       #
# --------------------------------------------------------------------------- #

SCRIPT_DIR = Path(__file__).parent.resolve()

# Reference artefacts (read-only — never written)
SEI_ADAPTER = SCRIPT_DIR / "sei_lora_adapter"
NCX_ADAPTER = SCRIPT_DIR / "ncx_lora_adapter"
REFERENCE_JOINT = SCRIPT_DIR / "ncx_asr_model" / "ncx-sei-joint-asr"  # XLS-R; do not touch

# New outputs go here
MIN_DIR = SCRIPT_DIR / "min_experiments"
JOINT_ADAPTER_DIR = MIN_DIR / "joint_adapter"   # trained joint adapter
EVALS_DIR = MIN_DIR / "evals"                   # per-adapter × test-set evaluations
SWEEPS_DIR = MIN_DIR / "tau_sweeps"             # τ-sweep outputs
B4_COMPAT_DIR = MIN_DIR / "b4_compat"           # rewritten CSVs for eval_morpheme_f1.py
SUMMARY_DIR = MIN_DIR / "summary"

LANGUAGES = ["sei", "ncx"]
TAU_VALUES = [0.0, 0.3, 0.5, 0.7, 1.0]

# --------------------------------------------------------------------------- #
# Data discovery                                                              #
# --------------------------------------------------------------------------- #

DATA_ROOT_TEMPLATES = [
    "/var/tmp/soobenve_audio/mdc_data/{lang}/extracted",
    "/part/01/Tmp/soobenve/nlp_project/mdc_data/{lang}/extracted",
    "/part/01/Tmp/soobenve/mdc_data/{lang}/extracted",
    str(Path.home() / "Downloads/nlp_project/mdc_data/{lang}/extracted"),
    "./mdc_data/{lang}/extracted",
]


def find_cv_root(lang: str, override: Optional[str] = None) -> Path:
    """Return the CV corpus root containing train.tsv / dev.tsv / test.tsv / clips/."""
    if override:
        p = Path(override.format(lang=lang) if "{lang}" in override else override) / lang / "extracted"
        if not p.exists():
            # maybe they gave us the extracted root directly
            p = Path(override.format(lang=lang) if "{lang}" in override else override)
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
        f"Cannot find Common Voice data for '{lang}'. "
        f"Tried: {[str(c) for c in candidates]}. Use --data_dir."
    )


# --------------------------------------------------------------------------- #
# Safety check — refuse to touch reference artefacts                          #
# --------------------------------------------------------------------------- #

def assert_safe_path(p: Path, purpose: str):
    """Refuse to write to any of the frozen reference artefact paths."""
    forbidden = [SEI_ADAPTER, NCX_ADAPTER, REFERENCE_JOINT]
    p_resolved = p.resolve()
    for f in forbidden:
        try:
            f_resolved = f.resolve()
        except Exception:
            continue
        # Refuse if equal or nested under a forbidden path
        if p_resolved == f_resolved:
            raise SystemExit(f"REFUSING to write to reference path {f} ({purpose})")
        try:
            p_resolved.relative_to(f_resolved)
            raise SystemExit(f"REFUSING to write inside reference path {f} ({purpose})")
        except ValueError:
            pass


# --------------------------------------------------------------------------- #
# Training machinery (borrowed from whisper_lora_train.py with patch)         #
# --------------------------------------------------------------------------- #

def train_joint_adapter(
    output_dir: Path,
    data_dir_override: Optional[str],
    per_device_batch: int,
    grad_accum: int,
    num_train_epochs: int,
    learning_rate: float,
    seed: int,
    no_wandb: bool,
):
    """Train a single LoRA adapter on concatenated sei+ncx training data.

    The processor is openai/whisper-small's multilingual tokenizer, which
    handles both languages in its subword inventory. We don't force any
    language token — the adapter learns to handle both by example.
    """
    import evaluate
    import numpy as np
    import torch
    from datasets import Dataset, DatasetDict, concatenate_datasets
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    assert_safe_path(output_dir, "joint training output")

    # PEFT + Whisper compatibility patch (from whisper_lora_train.py)
    _orig_fwd = WhisperForConditionalGeneration.forward
    _valid = set(inspect.signature(_orig_fwd).parameters.keys())

    def _patched_fwd(self, *args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in _valid}
        return _orig_fwd(self, *args, **kwargs)

    WhisperForConditionalGeneration.forward = _patched_fwd

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load TSVs for both languages
    import pandas as pd
    import librosa

    def _load_split(cv_root: Path, split: str) -> Optional[Dataset]:
        import csv as _csv
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
                "audio": df["audio_path"].tolist(),
                "sentence": df["sentence"].tolist(),
            })
            return ds
        return None

    print("\n[joint-train] Loading sei + ncx training data...")
    joint_splits = {"train": [], "validation": [], "test": []}
    for lang in LANGUAGES:
        cv_root = find_cv_root(lang, data_dir_override)
        print(f"  {lang} CV root: {cv_root}")
        for split in joint_splits:
            ds = _load_split(cv_root, split)
            if ds is not None:
                joint_splits[split].append(ds)
                print(f"    {split}: {len(ds)}")

    raw_datasets = DatasetDict({
        split: concatenate_datasets(ds_list).shuffle(seed=seed)
        for split, ds_list in joint_splits.items() if ds_list
    })
    for split, ds in raw_datasets.items():
        print(f"  joint {split}: {len(ds)}")

    # Processor
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", task="transcribe"
    )
    processor.tokenizer.set_prefix_tokens(language=None, task="transcribe")

    # Feature extraction
    def _prepare(batch):
        audio_path = batch["audio"]
        audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
        batch["input_features"] = processor.feature_extractor(
            audio_array, sampling_rate=sr
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    print("\n[joint-train] Encoding audio features...")
    encoded = raw_datasets.map(
        _prepare,
        remove_columns=raw_datasets["train"].column_names,
        num_proc=4,
        desc="encoding",
    )

    # Model + LoRA
    print("\n[joint-train] Loading openai/whisper-small + LoRA(r=32)...")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # Collator
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

    # Metrics
    wer_m = evaluate.load("wer", experiment_id=f"joint-{os.getpid()}-wer")
    cer_m = evaluate.load("cer", experiment_id=f"joint-{os.getpid()}-cer")

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
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )
            if labels is not None:
                inputs["labels"] = labels
            return loss, preds, labels

    if no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ.setdefault("WANDB_PROJECT", "shout-minimum")

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
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=128,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none" if no_wandb else "wandb",
        run_name="joint-sei-ncx",
        dataloader_num_workers=8,
        remove_unused_columns=False,
        seed=seed,
    )

    trainer = _Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded.get("validation"),
        data_collator=_Collator(processor=processor),
        compute_metrics=_metrics,
        processing_class=processor.feature_extractor,
    )

    print("\n[joint-train] Starting training...")
    trainer.train()

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    print(f"[joint-train] Adapter saved → {adapter_dir}")

    # Save training config for traceability
    with open(output_dir / "train_config.json", "w") as f:
        json.dump({
            "base_model": "openai/whisper-small",
            "lora_r": 32, "lora_alpha": 64, "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],
            "per_device_batch": per_device_batch,
            "grad_accum": grad_accum,
            "num_epochs": num_train_epochs,
            "learning_rate": learning_rate,
            "seed": seed,
            "training_data": "concat(sei_cv_train, ncx_cv_train) shuffled",
        }, f, indent=2)


# --------------------------------------------------------------------------- #
# Evaluation with per-word confidence extraction                              #
# --------------------------------------------------------------------------- #

def evaluate_adapter(
    adapter_path: Optional[Path],
    language: str,
    output_dir: Path,
    data_dir_override: Optional[str],
    batch_size: int = 16,
):
    """Evaluate an adapter (or zero-shot if adapter_path is None) on the
    given language's test set. Writes:
      - test_predictions.csv (language, reference, prediction)
      - test_tokens.jsonl    (per-word min_p for τ-sweep)
      - test_metrics.json    (WER, CER, clip count)
    """
    import evaluate
    import librosa
    import torch
    import torch.nn.functional as F
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    assert_safe_path(output_dir, f"evaluation of {adapter_path or 'zero-shot'} on {language}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[eval] Loading openai/whisper-small on {device}...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    if adapter_path is not None:
        from peft import PeftModel
        print(f"[eval] Loading adapter {adapter_path}...")
        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval().to(device)

    # Load test set
    cv_root = find_cv_root(language, data_dir_override)
    clips_dir = cv_root / "clips"
    import csv as _csv
    examples = []
    with open(cv_root / "test.tsv", newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f, delimiter="\t"):
            audio_path = clips_dir / row["path"]
            if audio_path.exists():
                sentence = row.get("sentence", row.get("text", "")).lower().strip()
                if sentence:
                    examples.append({"path": str(audio_path), "sentence": sentence})
    print(f"[eval] {len(examples)} test clips for {language}")

    references = [e["sentence"] for e in examples]
    audio_paths = [e["path"] for e in examples]

    # Inference with token probabilities
    predictions = []
    token_probs = []

    for i in range(0, len(audio_paths), batch_size):
        batch = audio_paths[i:i + batch_size]
        arrays = []
        for p in batch:
            try:
                arr, _ = librosa.load(p, sr=16000, mono=True)
                arrays.append(arr)
            except Exception as e:
                print(f"  warning: {p}: {e}")
                arrays.append(None)

        valid = [(j, a) for j, a in enumerate(arrays) if a is not None]
        batch_preds = [""] * len(arrays)
        batch_tokens = [[] for _ in arrays]

        if valid:
            idxs, arrs = zip(*valid)
            inputs = processor.feature_extractor(
                arrs, sampling_rate=16000, return_tensors="pt",
                padding="max_length",
            ).input_features.to(device)

            with torch.no_grad():
                gen_out = model.generate(
                    input_features=inputs, max_new_tokens=128,
                    language=None, task="transcribe",
                    output_scores=True, return_dict_in_generate=True,
                )

            sequences = gen_out.sequences
            scores = gen_out.scores
            prompt_len = sequences.shape[1] - len(scores)
            tokenizer = processor.tokenizer

            for bi, orig_idx in enumerate(idxs):
                gen_ids = sequences[bi, prompt_len:].tolist()
                token_ps = []
                for step, step_scores in enumerate(scores):
                    probs_step = F.softmax(step_scores[bi], dim=-1)
                    tid = gen_ids[step]
                    token_ps.append(float(probs_step[tid].item()))
                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                batch_preds[orig_idx] = text.lower().strip()

                words = []
                cur_word, cur_min_p = "", 1.0
                for tid, p in zip(gen_ids, token_ps):
                    if tid in tokenizer.all_special_ids:
                        continue
                    piece = tokenizer.decode([tid], skip_special_tokens=True)
                    if not piece:
                        continue
                    if piece.startswith(" ") and cur_word:
                        words.append({"word": cur_word.lower().strip(),
                                      "min_p": cur_min_p})
                        cur_word = piece.lstrip()
                        cur_min_p = p
                    else:
                        cur_word += piece
                        cur_min_p = min(cur_min_p, p)
                if cur_word.strip():
                    words.append({"word": cur_word.lower().strip(),
                                  "min_p": cur_min_p})
                batch_tokens[orig_idx] = words

        predictions.extend(batch_preds)
        token_probs.extend(batch_tokens)
        done = min(i + batch_size, len(audio_paths))
        print(f"  [{done}/{len(audio_paths)}]", end="\r")
    print()

    wer = evaluate.load("wer").compute(predictions=predictions, references=references)
    cer = evaluate.load("cer").compute(predictions=predictions, references=references)

    # Write outputs
    with open(output_dir / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=["language", "reference", "prediction"])
        w.writeheader()
        for ref, pred in zip(references, predictions):
            w.writerow({"language": language, "reference": ref, "prediction": pred})

    with open(output_dir / "test_tokens.jsonl", "w", encoding="utf-8") as f:
        for ref, pred, probs in zip(references, predictions, token_probs):
            f.write(json.dumps({
                "language": language,
                "reference": ref,
                "prediction": pred,
                "words": probs,
            }, ensure_ascii=False) + "\n")

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump({
            "adapter": str(adapter_path) if adapter_path else None,
            "language": language,
            "wer": wer, "cer": cer, "clips": len(examples),
        }, f, indent=2)

    print(f"[eval] WER {wer:.4f}  CER {cer:.4f}  → {output_dir}")
    return {"wer": wer, "cer": cer, "clips": len(examples)}


# --------------------------------------------------------------------------- #
# τ-sweep reconstruction (covers B4 and B4.5 in one pass)                     #
# --------------------------------------------------------------------------- #

def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                            prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def _closest_word(word: str, vocab: List[str], max_dist: int = 2) -> str:
    if word in vocab:
        return word
    best, best_d = word, max_dist + 1
    for v in vocab:
        if abs(len(v) - len(word)) > max_dist:
            continue
        d = _levenshtein(word, v)
        if d < best_d:
            best_d, best = d, v
    return best


def _simple_wer(preds: List[str], refs: List[str]) -> float:
    total_words, total_errors = 0, 0
    for pred, ref in zip(preds, refs):
        rw, pw = ref.split(), pred.split()
        n, m = len(rw), len(pw)
        total_words += n
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1): dp[i][0] = i
        for j in range(m + 1): dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1,
                               dp[i-1][j-1] + (rw[i-1] != pw[j-1]))
        total_errors += dp[n][m]
    return total_errors / total_words if total_words > 0 else 0.0


def _simple_cer(preds: List[str], refs: List[str]) -> float:
    return _simple_wer(
        [" ".join(list(p)) for p in preds],
        [" ".join(list(r)) for r in refs],
    )


def build_vocab_from_tsv(tsv_path: Path) -> List[str]:
    vocab = set()
    with open(tsv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            s = row.get("sentence", row.get("text", "")).lower().strip()
            vocab.update(s.split())
    return sorted(vocab)


def run_tau_sweep(
    tokens_jsonl: Path,
    language: str,
    vocab_list: List[str],
    output_dir: Path,
    tau_values: List[float] = TAU_VALUES,
    max_edit_dist: int = 2,
) -> Dict[str, Any]:
    """Sweep τ values for a single prediction set. Writes:
      - recon_tau{τ}.csv   for each τ  (schema matches reconstruct_b4.py output
                                        when τ=1.0, so eval_morpheme_f1 works)
      - threshold_sweep.json   with the summary curve
    """
    assert_safe_path(output_dir, f"τ-sweep on {language}")
    output_dir.mkdir(parents=True, exist_ok=True)

    clips = []
    with open(tokens_jsonl) as f:
        for line in f:
            clips.append(json.loads(line))
    references = [c["reference"] for c in clips]
    originals = [c["prediction"] for c in clips]

    wer_base = _simple_wer(originals, references)
    cer_base = _simple_cer(originals, references)

    sweep_rows = []
    print(f"  baseline: WER {wer_base:.4f} CER {cer_base:.4f}")
    for tau in tau_values:
        reconstructed = []
        n_rewritten = 0
        for clip in clips:
            out_words = []
            for w in clip["words"]:
                word, min_p = w["word"], w["min_p"]
                if min_p >= tau:
                    out_words.append(word)
                else:
                    new_w = _closest_word(word, vocab_list, max_edit_dist)
                    if new_w != word:
                        n_rewritten += 1
                    out_words.append(new_w)
            reconstructed.append(" ".join(out_words))

        wer_after = _simple_wer(reconstructed, references)
        cer_after = _simple_cer(reconstructed, references)
        n_cand = sum(1 for c in clips for w in c["words"] if w["min_p"] < tau)
        n_words = sum(len(c["words"]) for c in clips)

        print(f"  τ={tau:.2f}  WER {wer_after:.4f} (Δ{wer_after-wer_base:+.4f})  "
              f"CER {cer_after:.4f} (Δ{cer_after-cer_base:+.4f})  "
              f"candidates {n_cand}/{n_words}  rewritten {n_rewritten}")

        # Schema identical to reconstruct_b4.py output so that eval_morpheme_f1.py
        # reads it without modification. Write every τ — we'll pick τ=1.0 later
        # for the morpheme F1 evaluation.
        csv_path = output_dir / f"recon_tau{tau:.2f}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["language", "reference",
                                              "prediction", "corrected"])
            w.writeheader()
            for ref, orig, corr in zip(references, originals, reconstructed):
                w.writerow({
                    "language": language, "reference": ref,
                    "prediction": orig, "corrected": corr,
                })

        sweep_rows.append({
            "threshold": tau,
            "wer_before": wer_base, "wer_after": wer_after,
            "wer_delta": wer_after - wer_base,
            "cer_before": cer_base, "cer_after": cer_after,
            "cer_delta": cer_after - cer_base,
            "n_candidates": n_cand, "n_words": n_words,
            "n_rewritten": n_rewritten,
        })

    summary = {
        "language": language,
        "tokens_file": str(tokens_jsonl),
        "vocab_size": len(vocab_list),
        "baseline_wer": wer_base, "baseline_cer": cer_base,
        "max_edit_dist": max_edit_dist,
        "sweep": sweep_rows,
    }
    with open(output_dir / "threshold_sweep.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# --------------------------------------------------------------------------- #
# Morpheme F1 (invokes existing eval_morpheme_f1.py conventions)              #
# --------------------------------------------------------------------------- #

def run_morpheme_f1_on_tau1(sweep_dir: Path, language: str, output_dir: Path):
    """Copy the τ=1.0 CSV into the schema eval_morpheme_f1.py expects, then
    run that existing script. Writes morpheme_f1_results.json."""
    import shutil

    assert_safe_path(output_dir, f"morpheme F1 output for {language}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # The B4 CSV is literally the τ=1.0 sweep output
    src = sweep_dir / "recon_tau1.00.csv"
    if not src.exists():
        print(f"  [morpheme-F1] missing source: {src}, skipping")
        return None
    dst = output_dir / f"b4_{language}_reconstruction.csv"
    shutil.copy2(src, dst)

    # Invoke the existing eval_morpheme_f1.py if available
    em_script = SCRIPT_DIR / "eval_morpheme_f1.py"
    if em_script.exists():
        import subprocess
        result = subprocess.run(
            [sys.executable, str(em_script),
             "--lang", language,
             "--b4_dir", str(output_dir),
             "--output", str(output_dir / "morpheme_f1_results.json")],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  [morpheme-F1] eval_morpheme_f1.py failed:\n{result.stderr}")
            return None
        print(result.stdout.strip().split("\n")[-3:])  # last few lines
        json_path = output_dir / "morpheme_f1_results.json"
        if json_path.exists():
            with open(json_path) as f:
                return json.load(f).get(language)
    return None


# --------------------------------------------------------------------------- #
# Orchestrator                                                                 #
# --------------------------------------------------------------------------- #

# Every adapter × test-set pair we want to produce
EVAL_PAIRS = [
    # (slug for directory, adapter_path or None, language, description)
    ("sei_adapter_on_sei",   SEI_ADAPTER,       "sei", "B2 diagonal (reference)"),
    ("ncx_adapter_on_ncx",   NCX_ADAPTER,       "ncx", "B2 diagonal (missing metrics)"),
    ("joint_adapter_on_sei", JOINT_ADAPTER_DIR / "adapter", "sei", "B3 on sei"),
    ("joint_adapter_on_ncx", JOINT_ADAPTER_DIR / "adapter", "ncx", "B3 on ncx"),
    ("sei_adapter_on_ncx",   SEI_ADAPTER,       "ncx", "off-diagonal: sei→ncx"),
    ("ncx_adapter_on_sei",   NCX_ADAPTER,       "sei", "off-diagonal: ncx→sei"),
]


def step_train_joint(args):
    """Train the joint adapter if it doesn't already exist."""
    adapter_path = JOINT_ADAPTER_DIR / "adapter"
    config_marker = JOINT_ADAPTER_DIR / "train_config.json"
    if adapter_path.exists() and config_marker.exists():
        print(f"\n[joint-train] SKIP — adapter already at {adapter_path}")
        return
    if args.skip_train:
        print(f"\n[joint-train] SKIP — --skip-train and no existing adapter at {adapter_path}")
        print("             (later steps that need the joint adapter will be skipped too)")
        return
    train_joint_adapter(
        output_dir=JOINT_ADAPTER_DIR,
        data_dir_override=args.data_dir,
        per_device_batch=args.per_device_batch,
        grad_accum=args.grad_accum,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        no_wandb=args.no_wandb,
    )


def step_evaluate_all(args):
    """Evaluate every adapter × test-set pair, skipping any already done."""
    results = {}
    for slug, adapter, lang, desc in EVAL_PAIRS:
        out_dir = EVALS_DIR / slug
        metrics_file = out_dir / "test_metrics.json"

        # Skip if complete
        if metrics_file.exists():
            with open(metrics_file) as f:
                results[slug] = json.load(f)
            print(f"\n[eval] SKIP {slug} — already at {out_dir}")
            continue

        # Skip if adapter doesn't exist (e.g. joint not trained yet)
        if adapter is not None and not adapter.exists():
            print(f"\n[eval] SKIP {slug} — adapter not at {adapter}")
            continue

        print(f"\n[eval] {slug} ({desc})")
        try:
            metrics = evaluate_adapter(
                adapter_path=adapter, language=lang,
                output_dir=out_dir,
                data_dir_override=args.data_dir,
                batch_size=args.eval_batch_size,
            )
            results[slug] = metrics
        except Exception as e:
            print(f"[eval] FAIL {slug}: {e}")
            results[slug] = {"error": str(e)}
    return results


def step_tau_sweeps(args):
    """For every completed eval, run the τ-sweep and morpheme F1 at τ=1.0."""
    # Build vocab once per language
    vocabs = {}
    for lang in LANGUAGES:
        try:
            cv_root = find_cv_root(lang, args.data_dir)
        except FileNotFoundError:
            print(f"[sweep] skipping vocab for {lang} — no data root")
            continue
        # Prefer train.tsv, fall back to validated.tsv
        tsv = cv_root / "train.tsv"
        if not tsv.exists():
            tsv = cv_root / "validated.tsv"
        if not tsv.exists():
            print(f"[sweep] no train/validated.tsv for {lang}, skipping vocab")
            continue
        vocabs[lang] = build_vocab_from_tsv(tsv)
        print(f"[sweep] vocab[{lang}]: {len(vocabs[lang])} words from {tsv.name}")

    sweeps = {}
    morpheme = {}
    for slug, _, lang, _ in EVAL_PAIRS:
        eval_dir = EVALS_DIR / slug
        tokens_file = eval_dir / "test_tokens.jsonl"
        if not tokens_file.exists():
            continue
        if lang not in vocabs:
            print(f"[sweep] SKIP {slug} — no vocab for {lang}")
            continue

        sweep_dir = SWEEPS_DIR / slug
        summary_file = sweep_dir / "threshold_sweep.json"

        if summary_file.exists():
            with open(summary_file) as f:
                sweeps[slug] = json.load(f)
            print(f"\n[sweep] SKIP {slug} — already at {sweep_dir}")
        else:
            print(f"\n[sweep] {slug}")
            sweeps[slug] = run_tau_sweep(
                tokens_jsonl=tokens_file, language=lang,
                vocab_list=vocabs[lang],
                output_dir=sweep_dir,
                tau_values=args.tau_values,
                max_edit_dist=args.max_edit_dist,
            )

        # Morpheme F1 on τ=1.0 (the B4 equivalent)
        mf1_dir = B4_COMPAT_DIR / slug
        mf1_file = mf1_dir / "morpheme_f1_results.json"
        if mf1_file.exists():
            with open(mf1_file) as f:
                morpheme[slug] = json.load(f).get(lang)
            print(f"[morpheme-F1] SKIP {slug} — already computed")
        else:
            print(f"[morpheme-F1] {slug}")
            morpheme[slug] = run_morpheme_f1_on_tau1(sweep_dir, lang, mf1_dir)

    return sweeps, morpheme


def step_emit_summary(eval_results, sweeps, morpheme):
    """Write the consolidated summary table."""
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    # CSV: one row per adapter×test pair, showing baseline + every τ
    rows = []
    for slug, _, lang, desc in EVAL_PAIRS:
        row = {"slug": slug, "description": desc, "language": lang}
        ev = eval_results.get(slug)
        if ev and "wer" in ev:
            row["B2_wer"] = round(ev["wer"], 4)
            row["B2_cer"] = round(ev["cer"], 4)
            row["clips"] = ev.get("clips", "")
        sw = sweeps.get(slug)
        if sw and "sweep" in sw:
            for s in sw["sweep"]:
                tau = s["threshold"]
                row[f"tau{tau:.2f}_wer"] = round(s["wer_after"], 4)
                row[f"tau{tau:.2f}_delta"] = round(s["wer_delta"], 4)
        mf1 = morpheme.get(slug)
        if mf1:
            row["morpheme_f1_before"] = round(mf1.get("token_f1_before", 0.0), 4)
            row["morpheme_f1_after"] = round(mf1.get("token_f1_after", 0.0), 4)
            row["morpheme_f1_delta"] = round(mf1.get("token_f1_delta", 0.0), 4)
        rows.append(row)

    # Collect all column names seen
    all_cols = []
    for r in rows:
        for k in r.keys():
            if k not in all_cols:
                all_cols.append(k)

    summary_csv = SUMMARY_DIR / "summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in all_cols})

    # Paper-ready markdown
    md_lines = ["# Shout — minimum experiment suite results\n"]
    md_lines.append("## Results grid (WER)\n")
    md_lines.append("| Configuration | sei test | ncx test |")
    md_lines.append("|---|---|---|")

    def wer_for(slug):
        r = eval_results.get(slug)
        return f"{r['wer']:.4f}" if r and "wer" in r else "—"

    md_lines.append(f"| sei adapter | {wer_for('sei_adapter_on_sei')} | {wer_for('sei_adapter_on_ncx')} |")
    md_lines.append(f"| ncx adapter | {wer_for('ncx_adapter_on_sei')} | {wer_for('ncx_adapter_on_ncx')} |")
    md_lines.append(f"| joint adapter | {wer_for('joint_adapter_on_sei')} | {wer_for('joint_adapter_on_ncx')} |")
    md_lines.append("")

    md_lines.append("## τ-sweep (B4/B4.5 unified)\n")
    md_lines.append("| Config | τ=0.0 | τ=0.3 | τ=0.5 | τ=0.7 | τ=1.0 |")
    md_lines.append("|---|---|---|---|---|---|")
    md_lines.append("| | (passthrough) | | | | (B4 always-on) |")
    for slug, _, lang, desc in EVAL_PAIRS:
        sw = sweeps.get(slug)
        if not sw or "sweep" not in sw:
            continue
        cells = [f"**{slug}** ({lang})"]
        for s in sw["sweep"]:
            cells.append(f"{s['wer_after']:.4f}")
        md_lines.append("| " + " | ".join(cells) + " |")
    md_lines.append("")

    md_lines.append("## Morpheme F1 (at τ=1.0, i.e. B4 always-on)\n")
    md_lines.append("| Config | Token F1 before | Token F1 after | Δ |")
    md_lines.append("|---|---|---|---|")
    for slug, _, lang, desc in EVAL_PAIRS:
        mf = morpheme.get(slug)
        if not mf:
            continue
        md_lines.append(
            f"| **{slug}** ({lang}) | {mf.get('token_f1_before', 0):.4f} | "
            f"{mf.get('token_f1_after', 0):.4f} | "
            f"{mf.get('token_f1_delta', 0):+.4f} |"
        )

    md_path = SUMMARY_DIR / "summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # JSON dump of everything
    with open(SUMMARY_DIR / "summary.json", "w") as f:
        json.dump({
            "eval_results": eval_results,
            "tau_sweeps": sweeps,
            "morpheme_f1": morpheme,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Summary written:")
    print(f"  {summary_csv}")
    print(f"  {md_path}")
    print(f"  {SUMMARY_DIR / 'summary.json'}")
    print(f"{'='*60}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Shout — minimum experiment suite (single-file runner)"
    )
    parser.add_argument("--only", choices=["train", "eval", "tau_sweep", "summary", "all"],
                        default="all",
                        help="Run only a specific step. Default: all.")
    parser.add_argument("--skip-train", action="store_true",
                        help="Never train the joint adapter; use existing if present.")
    parser.add_argument("--data_dir", default=None,
                        help="Override CV data root. Use {lang} for substitution.")

    # Training knobs (applied only when the joint is being trained)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_device_batch", type=int, default=16,
                        help="A100 80GB handles 16 for whisper-small.")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--no_wandb", action="store_true")

    # Evaluation / sweep knobs
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--tau_values", type=float, nargs="+",
                        default=TAU_VALUES)
    parser.add_argument("--max_edit_dist", type=int, default=2)

    args = parser.parse_args()

    # Make sure output roots exist
    MIN_DIR.mkdir(parents=True, exist_ok=True)

    # Sanity banner
    print(f"{'='*60}")
    print(f"  Shout minimum experiment suite")
    print(f"  Step: {args.only}")
    print(f"  Outputs under: {MIN_DIR}")
    print(f"  Reference artefacts (read-only):")
    print(f"    {SEI_ADAPTER}  (exists={SEI_ADAPTER.exists()})")
    print(f"    {NCX_ADAPTER}  (exists={NCX_ADAPTER.exists()})")
    print(f"{'='*60}")

    # Verify adapters we depend on exist
    if not SEI_ADAPTER.exists() or not NCX_ADAPTER.exists():
        print("\nWARNING: reference adapters missing. Some evaluations will be skipped.")

    eval_results, sweeps, morpheme = {}, {}, {}

    if args.only in ("train", "all"):
        step_train_joint(args)

    if args.only in ("eval", "all"):
        eval_results = step_evaluate_all(args)

    if args.only in ("tau_sweep", "all"):
        # We need eval results to run sweeps; load them if they exist
        if not eval_results:
            for slug, _, _, _ in EVAL_PAIRS:
                mf = EVALS_DIR / slug / "test_metrics.json"
                if mf.exists():
                    with open(mf) as f:
                        eval_results[slug] = json.load(f)
        sweeps, morpheme = step_tau_sweeps(args)

    if args.only in ("summary", "all"):
        # Load from disk if not in-memory
        if not eval_results:
            for slug, _, _, _ in EVAL_PAIRS:
                mf = EVALS_DIR / slug / "test_metrics.json"
                if mf.exists():
                    with open(mf) as f:
                        eval_results[slug] = json.load(f)
        if not sweeps:
            for slug, _, _, _ in EVAL_PAIRS:
                sf = SWEEPS_DIR / slug / "threshold_sweep.json"
                if sf.exists():
                    with open(sf) as f:
                        sweeps[slug] = json.load(f)
        if not morpheme:
            for slug, _, lang, _ in EVAL_PAIRS:
                mf = B4_COMPAT_DIR / slug / "morpheme_f1_results.json"
                if mf.exists():
                    with open(mf) as f:
                        morpheme[slug] = json.load(f).get(lang)
        step_emit_summary(eval_results, sweeps, morpheme)

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
