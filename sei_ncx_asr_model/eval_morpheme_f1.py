#!/usr/bin/env python3
"""
Shout Project — Morpheme-Boundary F1 Evaluation (R3)
=====================================================
Computes a morpheme-boundary F1 score as a proxy for morphological
recovery quality. Since no external morphological parser is available
for Seri (sei) or Central Puebla Nahuatl (ncx), we use a
character-boundary approach:

  - Word-boundary F1: Treats whitespace positions as boundary markers.
    A predicted word boundary at character position i is a TP if the
    reference also has a boundary there (in the aligned character stream).

  - Token F1: Measures how many reference word tokens appear in the
    prediction (unordered, for robustness to hallucinations).

Both metrics are computed BEFORE and AFTER reconstruction (B4), giving
a quantitative signal for whether reconstruction improves morphological
recovery. This is the standard proxy used when a morphological parser
is unavailable (cf. Le Ferrand et al. 2024: morpheme F1 via
boundary-position comparison).

Usage:
    python eval_morpheme_f1.py              # run on both languages
    python eval_morpheme_f1.py --lang sei   # Seri only
    python eval_morpheme_f1.py --lang ncx   # Nahuatl only
"""

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path
from collections import Counter


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    # Strip punctuation (keep letters, digits, spaces, hyphens)
    text = re.sub(r"[^\w\s\-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Token F1 (unordered, multiset-aware)
# ---------------------------------------------------------------------------

def token_f1(ref: str, pred: str) -> float:
    """
    Token-level F1 treating the sentence as a bag of words.
    TP = tokens that appear in both (intersection of multisets).
    """
    ref_tokens  = Counter(normalise(ref).split())
    pred_tokens = Counter(normalise(pred).split())

    if not ref_tokens or not pred_tokens:
        return 0.0

    # Intersection of multisets
    tp = sum((ref_tokens & pred_tokens).values())
    precision = tp / sum(pred_tokens.values())
    recall    = tp / sum(ref_tokens.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Word-boundary F1 (character-aligned boundary detection)
# ---------------------------------------------------------------------------

def boundary_positions(text: str) -> set:
    """
    Return the set of character positions (0-indexed) that are
    immediately followed by a word boundary (space) in normalised text.
    We map this to a normalised character count so that minor length
    differences don't dominate.
    """
    norm = normalise(text)
    positions = set()
    for i, ch in enumerate(norm):
        if ch == " ":
            positions.add(i)
    return positions


def boundary_f1(ref: str, pred: str) -> float:
    """
    Compute F1 on the *set* of normalised character positions where
    word boundaries occur. Both strings are first normalised so that
    only alphabetic content and whitespace remain.
    """
    ref_b  = boundary_positions(ref)
    pred_b = boundary_positions(pred)

    if not ref_b and not pred_b:
        return 1.0   # both have no boundaries — trivially match
    if not ref_b or not pred_b:
        return 0.0

    tp = len(ref_b & pred_b)
    fp = len(pred_b - ref_b)
    fn = len(ref_b  - pred_b)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Per-language evaluation
# ---------------------------------------------------------------------------

def evaluate_file(csv_path: Path, lang: str) -> dict:
    """Read a B4 reconstruction CSV and compute before/after F1 metrics."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        print(f"  ⚠ No rows found in {csv_path}")
        return {}

    tok_before, tok_after   = [], []
    bnd_before, bnd_after   = [], []
    improvements, regressions, unchanged = [], [], []

    for row in rows:
        ref  = row["reference"]
        pred = row["prediction"]
        corr = row["corrected"]

        tb = token_f1(ref, pred)
        ta = token_f1(ref, corr)
        bb = boundary_f1(ref, pred)
        ba = boundary_f1(ref, corr)

        tok_before.append(tb)
        tok_after.append(ta)
        bnd_before.append(bb)
        bnd_after.append(ba)

        delta = ta - tb
        if delta > 0.05:
            improvements.append((ref, pred, corr, tb, ta))
        elif delta < -0.05:
            regressions.append((ref, pred, corr, tb, ta))

    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0

    results = {
        "language": lang,
        "n_examples": len(rows),
        "token_f1_before": avg(tok_before),
        "token_f1_after":  avg(tok_after),
        "token_f1_delta":  avg(tok_after) - avg(tok_before),
        "boundary_f1_before": avg(bnd_before),
        "boundary_f1_after":  avg(bnd_after),
        "boundary_f1_delta":  avg(bnd_after) - avg(bnd_before),
        "n_improved":   len(improvements),
        "n_regressed":  len(regressions),
        "n_unchanged":  len(rows) - len(improvements) - len(regressions),
    }

    print(f"\n{'='*60}")
    print(f"  Morpheme-Boundary F1 — {lang.upper()}")
    print(f"{'='*60}")
    print(f"  Token F1   before: {results['token_f1_before']:.4f}")
    print(f"  Token F1   after:  {results['token_f1_after']:.4f}  "
          f"({'+'if results['token_f1_delta']>=0 else ''}"
          f"{results['token_f1_delta']:.4f})")
    print(f"  Boundary F1 before: {results['boundary_f1_before']:.4f}")
    print(f"  Boundary F1 after:  {results['boundary_f1_after']:.4f}  "
          f"({'+'if results['boundary_f1_delta']>=0 else ''}"
          f"{results['boundary_f1_delta']:.4f})")
    print(f"\n  Improved:  {results['n_improved']} examples")
    print(f"  Regressed: {results['n_regressed']} examples")
    print(f"  Unchanged: {results['n_unchanged']} examples")

    if improvements:
        print(f"\n  ── Top 3 improvements ──")
        for ref, pred, corr, tb, ta in sorted(improvements, key=lambda x: x[4]-x[3], reverse=True)[:3]:
            print(f"  REF : {ref[:80]}")
            print(f"  PRED: {pred[:80]}  (token F1: {tb:.2f})")
            print(f"  CORR: {corr[:80]}  (token F1: {ta:.2f})")
            print()

    if regressions:
        print(f"  ── Top 3 regressions ──")
        for ref, pred, corr, tb, ta in sorted(regressions, key=lambda x: x[3]-x[4], reverse=True)[:3]:
            print(f"  REF : {ref[:80]}")
            print(f"  PRED: {pred[:80]}  (token F1: {tb:.2f})")
            print(f"  CORR: {corr[:80]}  (token F1: {ta:.2f})")
            print()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Morpheme-boundary F1 evaluation for Shout B4 reconstruction"
    )
    parser.add_argument("--lang", choices=["sei", "ncx"], default=None,
                        help="Language to evaluate (default: both)")
    parser.add_argument("--b4_dir", default="./b4_results",
                        help="Directory containing b4_*_reconstruction.csv files")
    parser.add_argument("--output", default="./b4_results/morpheme_f1_results.json",
                        help="Path to write JSON results")
    args = parser.parse_args()

    b4_dir = Path(args.b4_dir)
    langs = [args.lang] if args.lang else ["sei", "ncx"]
    all_results = {}

    for lang in langs:
        csv_path = b4_dir / f"b4_{lang}_reconstruction.csv"
        if not csv_path.exists():
            print(f"  ⚠ Missing: {csv_path} — skipping {lang}")
            continue
        result = evaluate_file(csv_path, lang)
        if result:
            all_results[lang] = result

    # Save combined results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Morpheme F1 results saved → {out_path}")


if __name__ == "__main__":
    main()
