#!/usr/bin/env python3
"""
Shout — B4.5 threshold sweep
==============================

Runs uncertainty-triggered lexicon reconstruction at multiple confidence
thresholds, producing a curve of WER vs threshold. This is the compute for
the B4.5 paper section.

Threshold semantics (matches reconstruct.js:reconstructTokens):
  threshold = 1.0  → always rewrite (equivalent to always-on B4)
  threshold = 0.5  → rewrite words whose min token p < 0.5
  threshold = 0.0  → never rewrite (equivalent to B1/B2 raw output)

Interpretation of the resulting curve:
  Monotonic decreasing WER with threshold     → lookup alone drives gains;
                                                  trigger adds nothing
  Inverted-U (best at intermediate threshold) → trigger matters: aggressive
                                                  rewriting harms confident tokens
  Flat                                         → BK-tree is conservative enough
                                                  on its own; threshold doesn't
                                                  change behaviour much

Input:
  A test_tokens.jsonl produced by eval_adapter.py (one line per clip with
  per-word min-p data) — plus a vocab list (training-set words).

Usage:
  python reconstruct_threshold_sweep.py \\
    --tokens experiments/seed_replicates/ncx_seed42/test_tokens.jsonl \\
    --train_tsv /path/to/cv/ncx/train.tsv \\
    --language ncx \\
    --output_dir experiments/b45_sweep/ncx_seed42 \\
    --thresholds 0.0 0.3 0.5 0.7 1.0
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(
                prev[j + 1] + 1,
                curr[j] + 1,
                prev[j] + (ca != cb),
            ))
        prev = curr
    return prev[-1]


def closest_word(word: str, vocab: List[str], max_dist: int = 2) -> str:
    if word in vocab:
        return word
    best, best_d = word, max_dist + 1
    for v in vocab:
        if abs(len(v) - len(word)) > max_dist:
            continue
        d = levenshtein(word, v)
        if d < best_d:
            best_d, best = d, v
    return best


def simple_wer(predictions: List[str], references: List[str]) -> float:
    total_words, total_errors = 0, 0
    for pred, ref in zip(predictions, references):
        ref_words, pred_words = ref.split(), pred.split()
        n, m = len(ref_words), len(pred_words)
        total_words += n
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


def simple_cer(predictions: List[str], references: List[str]) -> float:
    # Char-level WER
    return simple_wer(
        [" ".join(list(p)) for p in predictions],
        [" ".join(list(r)) for r in references],
    )


def build_vocab_from_tsv(tsv_path: Path) -> set:
    vocab = set()
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sentence = row.get("sentence", row.get("text", "")).lower().strip()
            vocab.update(sentence.split())
    return vocab


def reconstruct_with_threshold(
    words_with_probs: List[dict],
    vocab_list: List[str],
    threshold: float,
    max_edit_dist: int = 2,
):
    """
    For each word:
      - if min_p >= threshold:   keep as-is (confident enough)
      - else:                    route through BK-tree lookup
    Returns (rewritten_sentence, num_rewritten).
    """
    out_words = []
    num_rewritten = 0
    for w in words_with_probs:
        word = w["word"]
        min_p = w["min_p"]
        if min_p >= threshold:
            out_words.append(word)
        else:
            new_word = closest_word(word, vocab_list, max_edit_dist)
            if new_word != word:
                num_rewritten += 1
            out_words.append(new_word)
    return " ".join(out_words), num_rewritten


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", required=True,
                   help="test_tokens.jsonl from eval_adapter.py")
    p.add_argument("--train_tsv", required=True,
                   help="CV train.tsv to build vocab from")
    p.add_argument("--language", required=True, choices=["sei", "ncx"])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[0.0, 0.3, 0.5, 0.7, 1.0])
    p.add_argument("--max_edit_dist", type=int, default=2)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load vocab
    vocab = build_vocab_from_tsv(Path(args.train_tsv))
    vocab_list = sorted(vocab)
    print(f"Vocab: {len(vocab_list)} unique words")

    # Load per-clip token data
    clips = []
    with open(args.tokens) as f:
        for line in f:
            clips.append(json.loads(line))
    print(f"Clips: {len(clips)}")

    references = [c["reference"] for c in clips]
    originals = [c["prediction"] for c in clips]
    wer_before = simple_wer(originals, references)
    cer_before = simple_cer(originals, references)

    print(f"\nBaseline (no reconstruction): WER {wer_before:.4f}  CER {cer_before:.4f}")
    print(f"\nSweeping thresholds: {args.thresholds}\n")

    sweep_results = []
    for tau in args.thresholds:
        reconstructed = []
        total_rewritten = 0
        for clip in clips:
            new_sent, n_rewritten = reconstruct_with_threshold(
                clip["words"], vocab_list, tau, args.max_edit_dist
            )
            reconstructed.append(new_sent)
            total_rewritten += n_rewritten

        wer_after = simple_wer(reconstructed, references)
        cer_after = simple_cer(reconstructed, references)
        n_candidates = sum(1 for c in clips for w in c["words"] if w["min_p"] < tau)
        n_words = sum(len(c["words"]) for c in clips)

        print(f"  τ={tau:.2f}  "
              f"WER {wer_after:.4f} (Δ{wer_after-wer_before:+.4f})  "
              f"CER {cer_after:.4f} (Δ{cer_after-cer_before:+.4f})  "
              f"candidates {n_candidates}/{n_words} ({100*n_candidates/max(1,n_words):.1f}%)  "
              f"rewritten {total_rewritten}")

        # Save per-threshold predictions
        th_csv = out_dir / f"recon_tau{tau:.2f}.csv"
        with open(th_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "language", "reference", "prediction", "corrected"
            ])
            w.writeheader()
            for ref, orig, corr in zip(references, originals, reconstructed):
                w.writerow({
                    "language": args.language,
                    "reference": ref,
                    "prediction": orig,
                    "corrected": corr,
                })

        sweep_results.append({
            "threshold": tau,
            "wer_before": wer_before, "wer_after": wer_after,
            "wer_delta": wer_after - wer_before,
            "cer_before": cer_before, "cer_after": cer_after,
            "cer_delta": cer_after - cer_before,
            "n_candidates": n_candidates,
            "n_words": n_words,
            "n_rewritten": total_rewritten,
            "predictions_csv": str(th_csv),
        })

    # Save summary
    summary = {
        "language": args.language,
        "source_tokens": str(args.tokens),
        "vocab_size": len(vocab_list),
        "baseline_wer": wer_before,
        "baseline_cer": cer_before,
        "max_edit_dist": args.max_edit_dist,
        "sweep": sweep_results,
    }
    summary_path = out_dir / "threshold_sweep.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
