#!/usr/bin/env python3
"""
Shout Project — Baseline B1: Whisper Zero-Shot Evaluation
==========================================================
Evaluates openai/whisper-small on Seri and/or Nahuatl test sets
with NO fine-tuning. This establishes the zero-shot baseline and
confirms that LoRA adaptation (B2) actually helps.

Usage:
    python eval_b1_zero_shot.py --language sei
    python eval_b1_zero_shot.py --language ncx
    python eval_b1_zero_shot.py --language sei ncx   # both at once

Runs on CPU (slow) or GPU if available.
"""

import argparse
import csv
import json
from pathlib import Path

import evaluate
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #

BASE_MODEL_ID = "openai/whisper-small"

LANGUAGE_CONFIG = {
    "sei": {"name": "Seri"},
    "ncx": {"name": "Central Puebla Nahuatl"},
}

DATA_ROOTS = [
    "/var/tmp/soobenve_audio/mdc_data/{lang}/extracted",
    "/part/01/Tmp/soobenve/nlp_project/mdc_data/{lang}/extracted",
    str(Path.home() / "Downloads/nlp_project/mdc_data/{lang}/extracted"),
]


# ------------------------------------------------------------------ #
# Data helpers
# ------------------------------------------------------------------ #

def find_cv_test_tsv(language: str) -> tuple[Path, Path]:
    """Returns (cv_root, clips_dir) for the language test set."""
    for template in DATA_ROOTS:
        root = Path(template.format(lang=language))
        if root.exists():
            for clips in root.rglob("clips"):
                parent = clips.parent
                if (parent / "test.tsv").exists():
                    return parent, clips
    raise FileNotFoundError(
        f"Cannot find Common Voice data for '{language}'.\n"
        "Run the Wav2Vec2 notebook first to download it."
    )


def load_test_examples(cv_root: Path, clips_dir: Path) -> list[dict]:
    """Load test.tsv → list of {path, sentence}."""
    rows = []
    with open(cv_root / "test.tsv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            audio_path = clips_dir / row["path"]
            if audio_path.exists():
                sentence = row.get("sentence", row.get("text", "")).lower().strip()
                if sentence:
                    rows.append({"path": str(audio_path), "sentence": sentence})
    print(f"  Loaded {len(rows)} test examples")
    return rows


# ------------------------------------------------------------------ #
# Inference
# ------------------------------------------------------------------ #

def transcribe_batch(model, processor, audio_paths: list[str],
                     device: str, batch_size: int = 8) -> list[str]:
    """Run Whisper inference on a list of audio files."""
    import librosa

    results = []
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i : i + batch_size]
        arrays = []
        for p in batch_paths:
            try:
                arr, _ = librosa.load(p, sr=16000, mono=True)
                arrays.append(arr)
            except Exception as e:
                print(f"  Warning: could not load {p}: {e}")
                arrays.append(None)

        # Filter out None
        valid = [(j, a) for j, a in enumerate(arrays) if a is not None]
        batch_results = [""] * len(arrays)

        if valid:
            idxs, arrs = zip(*valid)
            inputs = processor.feature_extractor(
                arrs, sampling_rate=16000, return_tensors="pt",
                padding="max_length",  # always pad to 3000 mel frames
            ).input_features.to(device)

            with torch.no_grad():
                tokens = model.generate(
                    inputs,
                    max_new_tokens=128,
                    language=None,   # let Whisper auto-detect
                    task="transcribe",
                )
            decoded = processor.tokenizer.batch_decode(
                tokens, skip_special_tokens=True
            )
            for idx, text in zip(idxs, decoded):
                batch_results[idx] = text.lower().strip()

        results.extend(batch_results)

        done = min(i + batch_size, len(audio_paths))
        print(f"  [{done}/{len(audio_paths)}] transcribed...", end="\r")

    print()
    return results


# ------------------------------------------------------------------ #
# Main evaluation
# ------------------------------------------------------------------ #

def evaluate_language(language: str, model, processor, device: str,
                      output_dir: Path) -> dict:
    lang_name = LANGUAGE_CONFIG[language]["name"]
    print(f"\n{'='*55}")
    print(f"  B1 Zero-Shot | {lang_name} ({language})")
    print(f"{'='*55}")

    cv_root, clips_dir = find_cv_test_tsv(language)
    examples = load_test_examples(cv_root, clips_dir)

    audio_paths = [e["path"] for e in examples]
    references  = [e["sentence"] for e in examples]

    print(f"  Running zero-shot inference on {len(examples)} clips...")
    predictions = transcribe_batch(model, processor, audio_paths, device)

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)

    print(f"\n  ── Results ──")
    print(f"  WER : {wer:.4f}  ({wer*100:.1f}%)")
    print(f"  CER : {cer:.4f}  ({cer*100:.1f}%)")
    print(f"  Clips: {len(examples)}")

    # Save predictions CSV
    out_csv = output_dir / f"b1_zero_shot_{language}_predictions.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["language", "reference", "prediction"])
        writer.writeheader()
        for ref, pred in zip(references, predictions):
            writer.writerow({"language": language, "reference": ref, "prediction": pred})
    print(f"  Predictions saved → {out_csv}")

    metrics = {"language": language, "wer": wer, "cer": cer, "clips": len(examples)}
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Whisper zero-shot baseline (B1)")
    parser.add_argument("--language", nargs="+", required=True,
                        choices=list(LANGUAGE_CONFIG.keys()),
                        help="Language code(s): sei, ncx, or both")
    parser.add_argument("--output_dir", default="./b1_results",
                        help="Where to save results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {BASE_MODEL_ID} (zero-shot, no fine-tuning)...")
    print(f"Device: {device}")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID)
    model.config.forced_decoder_ids = None
    model.eval()
    model.to(device)

    all_metrics = []
    for lang in args.language:
        metrics = evaluate_language(lang, model, processor, device, output_dir)
        all_metrics.append(metrics)

    # Summary table
    print(f"\n{'='*55}")
    print("  B1 ZERO-SHOT SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Language':<12} {'WER':>8} {'CER':>8} {'Clips':>8}")
    print(f"  {'-'*40}")
    for m in all_metrics:
        print(f"  {m['language']:<12} {m['wer']:>8.4f} {m['cer']:>8.4f} {m['clips']:>8}")

    # Save JSON summary
    out_json = output_dir / "b1_zero_shot_metrics.json"
    with open(out_json, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Metrics saved → {out_json}")
    print("\nDone! These are your B1 baselines to compare against B2 (LoRA) and B3 (full FT).")


if __name__ == "__main__":
    main()
