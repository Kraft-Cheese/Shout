#!/usr/bin/env python3
"""
Shout Project — Baseline B4: Lexicon-Constrained Reconstruction
================================================================
A lightweight post-processing step that corrects ASR output by
replacing out-of-vocabulary words with the closest in-vocabulary
candidate (edit-distance). Runs entirely on CPU from existing CSVs.

Usage:
    # Run on B2 (Whisper+LoRA) predictions:
    python reconstruct_b4.py \
        --predictions sei_final_model/test_predictions.csv \
        --train_tsv /path/to/cv/sei/train.tsv \
        --language sei

    # Run on B1 zero-shot predictions:
    python reconstruct_b4.py \
        --predictions b1_results/b1_zero_shot_sei_predictions.csv \
        --train_tsv /path/to/cv/sei/train.tsv \
        --language sei

Output: augmented CSV with corrected predictions + before/after WER.
"""

import argparse
import csv
import json
from pathlib import Path


# ------------------------------------------------------------------ #
# Edit distance (Levenshtein)
# ------------------------------------------------------------------ #

def levenshtein(a: str, b: str) -> int:
    """Fast Levenshtein distance between two strings."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(
                prev[j + 1] + 1,   # deletion
                curr[j] + 1,       # insertion
                prev[j] + (ca != cb),  # substitution
            ))
        prev = curr
    return prev[-1]


def closest_word(word: str, vocab: list[str], max_dist: int = 2) -> str:
    """Return the closest word in vocab within max_dist edits, or original."""
    if word in vocab:
        return word
    best, best_d = word, max_dist + 1
    for v in vocab:
        # Quick length filter before full edit distance
        if abs(len(v) - len(word)) > max_dist:
            continue
        d = levenshtein(word, v)
        if d < best_d:
            best_d, best = d, v
    return best


# ------------------------------------------------------------------ #
# WER computation (without evaluate library — pure Python)
# ------------------------------------------------------------------ #

def simple_wer(predictions: list[str], references: list[str]) -> float:
    """Word error rate: (S + D + I) / N."""
    total_words, total_errors = 0, 0
    for pred, ref in zip(predictions, references):
        ref_words  = ref.split()
        pred_words = pred.split()
        n = len(ref_words)
        m = len(pred_words)
        total_words += n
        # DP edit distance at word level
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dp[i][j] = min(
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1,
                    dp[i-1][j-1] + (ref_words[i-1] != pred_words[j-1]),
                )
        total_errors += dp[n][m]
    return total_errors / total_words if total_words > 0 else 0.0


# ------------------------------------------------------------------ #
# Vocabulary building
# ------------------------------------------------------------------ #

def build_vocab_from_tsv(tsv_path: Path) -> set[str]:
    """Extract all unique words from the sentence column of a TSV."""
    vocab = set()
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sentence = row.get("sentence", row.get("text", "")).lower().strip()
            vocab.update(sentence.split())
    print(f"  Vocab size: {len(vocab)} unique words from {tsv_path.name}")
    return vocab


def build_vocab_from_references(references: list[str]) -> set[str]:
    """Build vocab from reference strings (fallback when no train TSV)."""
    vocab = set()
    for ref in references:
        vocab.update(ref.lower().split())
    print(f"  Vocab size: {len(vocab)} unique words from reference column")
    return vocab


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="B4 lexicon reconstruction")
    parser.add_argument("--predictions", required=True,
                        help="CSV with 'reference' and 'prediction' columns")
    parser.add_argument("--train_tsv", default=None,
                        help="Common Voice train.tsv to build vocabulary from "
                             "(if not given, uses reference column as lexicon)")
    parser.add_argument("--language", required=True, choices=["sei", "ncx"],
                        help="Language code for output naming")
    parser.add_argument("--max_edit_dist", type=int, default=2,
                        help="Max edit distance for correction (default: 2)")
    parser.add_argument("--output_dir", default="./b4_results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = Path(args.predictions)

    # ── Load predictions ──────────────────────────────────────────
    print(f"\nLoading predictions from {pred_path.name}...")
    references, predictions = [], []
    with open(pred_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            references.append(row["reference"].lower().strip())
            predictions.append(row["prediction"].lower().strip())
    print(f"  {len(references)} examples loaded")

    # ── Build vocabulary ──────────────────────────────────────────
    if args.train_tsv:
        vocab = build_vocab_from_tsv(Path(args.train_tsv))
    else:
        print("  No --train_tsv given; using reference column as vocabulary.")
        vocab = build_vocab_from_references(references)
    vocab_list = sorted(vocab)  # sort for determinism

    # ── Reconstruct ───────────────────────────────────────────────
    print(f"\nApplying lexicon correction (max_edit_dist={args.max_edit_dist})...")
    corrected = []
    num_words_changed = 0
    total_words = 0
    for pred in predictions:
        words = pred.split()
        total_words += len(words)
        fixed_words = []
        for w in words:
            c = closest_word(w, vocab_list, args.max_edit_dist)
            if c != w:
                num_words_changed += 1
            fixed_words.append(c)
        corrected.append(" ".join(fixed_words))

    # ── Metrics ───────────────────────────────────────────────────
    wer_before = simple_wer(predictions, references)
    wer_after  = simple_wer(corrected,   references)

    print(f"\n{'='*50}")
    print(f"  B4 RECONSTRUCTION RESULTS — {args.language.upper()}")
    print(f"{'='*50}")
    print(f"  WER before: {wer_before:.4f}  ({wer_before*100:.1f}%)")
    print(f"  WER after:  {wer_after:.4f}  ({wer_after*100:.1f}%)")
    delta = wer_after - wer_before
    sign = "+" if delta >= 0 else ""
    print(f"  Δ WER:      {sign}{delta:.4f}  ({sign}{delta*100:.1f}%)")
    print(f"  Words corrected: {num_words_changed}/{total_words}")

    # ── Qualitative examples ──────────────────────────────────────
    print(f"\n  {'─'*48}")
    print(f"  Sample corrections (showing changed predictions):")
    print(f"  {'─'*48}")
    shown = 0
    for ref, pred, corr in zip(references, predictions, corrected):
        if pred != corr and shown < 8:
            print(f"  REF : {ref}")
            print(f"  PRED: {pred}")
            print(f"  CORR: {corr}")
            print()
            shown += 1
    if shown == 0:
        print("  No words were changed (all predictions in vocabulary).")

    # ── Save output CSV ───────────────────────────────────────────
    out_csv = output_dir / f"b4_{args.language}_reconstruction.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["language", "reference", "prediction", "corrected"])
        writer.writeheader()
        for ref, pred, corr in zip(references, predictions, corrected):
            writer.writerow({
                "language": args.language,
                "reference": ref,
                "prediction": pred,
                "corrected": corr,
            })

    # ── Save JSON metrics ─────────────────────────────────────────
    metrics = {
        "language": args.language,
        "wer_before": wer_before,
        "wer_after": wer_after,
        "wer_delta": wer_after - wer_before,
        "words_corrected": num_words_changed,
        "total_words": total_words,
        "vocab_size": len(vocab),
        "max_edit_dist": args.max_edit_dist,
        "input_file": str(pred_path),
    }
    out_json = output_dir / f"b4_{args.language}_metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Results saved → {out_csv}")
    print(f"  Metrics saved → {out_json}")


if __name__ == "__main__":
    main()
