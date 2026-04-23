#!/usr/bin/env python3
"""
Shout — Adapter evaluation with per-word confidence extraction
================================================================

Evaluates a trained LoRA adapter on a language's test set and produces:

  1. test_predictions.csv     — same schema as eval_b1_zero_shot's output,
                                 drop-in for eval_morpheme_f1.py
  2. test_tokens.jsonl        — per-clip list of (word, min_token_p) for B4.5
                                 threshold sweeps
  3. test_metrics.json        — WER, CER, clip count

This is the missing producer the implementation plan flagged — the B4.5
ablation in reconstruct_b4.py can only run once per-word probabilities exist.

Usage:
  # Evaluate an adapter on its matched test set
  python eval_adapter.py --adapter experiments/seed_replicates/ncx_seed13/adapter \\
    --language ncx --output_dir experiments/seed_replicates/ncx_seed13

  # Off-diagonal: evaluate the sei adapter on the ncx test set
  python eval_adapter.py --adapter ./sei_lora_adapter \\
    --language ncx --output_dir analysis/sei_adapter_on_ncx

  # Zero-shot (no adapter)
  python eval_adapter.py --language ncx --output_dir analysis/zero_shot_ncx

  # Evaluate a non-default base model (for model-size sweep)
  python eval_adapter.py --adapter experiments/size_sweep/ncx_base/adapter \\
    --base_model base --language ncx \\
    --output_dir experiments/size_sweep/ncx_base
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import evaluate
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

BASE_MODEL_DEFAULTS = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
}

DATA_ROOTS = [
    "/var/tmp/soobenve_audio/mdc_data/{lang}/extracted",
    "/part/01/Tmp/soobenve/nlp_project/mdc_data/{lang}/extracted",
    str(Path.home() / "Downloads/nlp_project/mdc_data/{lang}/extracted"),
    "./mdc_data/{lang}/extracted",
]


def find_test_tsv(language: str):
    for template in DATA_ROOTS:
        root = Path(template.format(lang=language))
        if root.exists():
            for clips in root.rglob("clips"):
                parent = clips.parent
                if (parent / "test.tsv").exists():
                    return parent, clips
    raise FileNotFoundError(f"Cannot find Common Voice data for '{language}'")


def load_test_examples(cv_root: Path, clips_dir: Path):
    rows = []
    with open(cv_root / "test.tsv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            audio_path = clips_dir / row["path"]
            if audio_path.exists():
                sentence = row.get("sentence", row.get("text", "")).lower().strip()
                if sentence:
                    rows.append({"path": str(audio_path), "sentence": sentence})
    return rows


def transcribe_with_token_probs(model, processor, audio_paths, device, batch_size=8):
    """
    Returns parallel lists:
      predictions: list[str]
      token_probs: list[list[dict]] — per-clip list of {word, min_p}

    Word-level min-p is derived by:
      1. Running generate() with output_scores=True, return_dict_in_generate=True
      2. Converting logits → probs; for each generated token we take the probability
         of the chosen token
      3. Splitting the decoded text into words and mapping each word to the minimum
         probability among its constituent tokens. This is the quantity reconstruct_b4.py
         uses to decide whether to trigger reconstruction per word.
    """
    import librosa
    import torch.nn.functional as F

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
                    inputs,
                    max_new_tokens=128,
                    language=None,
                    task="transcribe",
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            # gen_out.sequences : (B, T_full)  — includes prompt tokens
            # gen_out.scores    : tuple of length T_new, each (B, vocab)
            sequences = gen_out.sequences  # (B, T_full)
            scores = gen_out.scores         # T_new tuples

            # The "new" tokens start at the end of the prompt; we find them by
            # comparing sequences width to scores length.
            prompt_len = sequences.shape[1] - len(scores)

            tokenizer = processor.tokenizer

            for bi, orig_idx in enumerate(idxs):
                gen_ids = sequences[bi, prompt_len:].tolist()
                # Per-step probability of the actually-chosen token
                token_ps = []
                for step, step_scores in enumerate(scores):
                    probs_step = F.softmax(step_scores[bi], dim=-1)
                    tid = gen_ids[step]
                    token_ps.append(float(probs_step[tid].item()))

                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                batch_preds[orig_idx] = text.lower().strip()

                # Group tokens into words by decoded text. Whisper BPE tokens
                # carry leading-space semantics: a space in the decoded string
                # starts a new word. We walk token-by-token, decoding each in
                # isolation (for boundary detection) while keeping the overall
                # decoded text canonical.
                words_with_probs = []
                cur_word = ""
                cur_min_p = 1.0
                for tid, p in zip(gen_ids, token_ps):
                    if tid in tokenizer.all_special_ids:
                        continue
                    piece = tokenizer.decode([tid], skip_special_tokens=True)
                    if not piece:
                        continue
                    if piece.startswith(" ") and cur_word:
                        words_with_probs.append({
                            "word": cur_word.lower().strip(),
                            "min_p": cur_min_p,
                        })
                        cur_word = piece.lstrip()
                        cur_min_p = p
                    else:
                        cur_word += piece
                        cur_min_p = min(cur_min_p, p)
                if cur_word.strip():
                    words_with_probs.append({
                        "word": cur_word.lower().strip(),
                        "min_p": cur_min_p,
                    })
                batch_tokens[orig_idx] = words_with_probs

        predictions.extend(batch_preds)
        token_probs.extend(batch_tokens)

        done = min(i + batch_size, len(audio_paths))
        print(f"  [{done}/{len(audio_paths)}]", end="\r")
    print()

    return predictions, token_probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--language", required=True, choices=["sei", "ncx"])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--adapter", default=None,
                   help="Path to PEFT adapter directory. Omit for zero-shot.")
    p.add_argument("--base_model", choices=list(BASE_MODEL_DEFAULTS.keys()),
                   default="small")
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_model_id = BASE_MODEL_DEFAULTS[args.base_model]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading base {base_model_id} on {device}...")
    processor = WhisperProcessor.from_pretrained(base_model_id)
    model = WhisperForConditionalGeneration.from_pretrained(base_model_id)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    if args.adapter:
        from peft import PeftModel
        print(f"Loading adapter from {args.adapter}...")
        model = PeftModel.from_pretrained(model, args.adapter)
    else:
        print("(no adapter — zero-shot evaluation)")

    model.eval().to(device)

    cv_root, clips_dir = find_test_tsv(args.language)
    examples = load_test_examples(cv_root, clips_dir)
    print(f"  loaded {len(examples)} test clips")

    audio_paths = [e["path"] for e in examples]
    references = [e["sentence"] for e in examples]

    print("Running inference with per-token probabilities...")
    predictions, token_probs = transcribe_with_token_probs(
        model, processor, audio_paths, device, args.batch_size
    )

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)

    # test_predictions.csv — schema identical to B1 output so
    # eval_morpheme_f1.py runs against it unchanged
    csv_path = out_dir / "test_predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["language", "reference", "prediction"])
        w.writeheader()
        for ref, pred in zip(references, predictions):
            w.writerow({"language": args.language, "reference": ref, "prediction": pred})
    print(f"Predictions  → {csv_path}")

    # test_tokens.jsonl — one line per clip, containing word-level probabilities.
    # Used by reconstruct_b4.py in threshold-sweep mode to decide which words
    # are candidates for correction.
    jsonl_path = out_dir / "test_tokens.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ref, pred, probs in zip(references, predictions, token_probs):
            f.write(json.dumps({
                "language": args.language,
                "reference": ref,
                "prediction": pred,
                "words": probs,
            }, ensure_ascii=False) + "\n")
    print(f"Token probs  → {jsonl_path}")

    # Metrics summary
    metrics = {
        "language": args.language,
        "adapter": args.adapter,
        "base_model": base_model_id,
        "wer": wer, "cer": cer,
        "clips": len(examples),
    }
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nWER {wer:.4f}  CER {cer:.4f}  clips {len(examples)}")


if __name__ == "__main__":
    main()
