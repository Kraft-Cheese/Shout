#!/usr/bin/env python3
"""
Shout — Missing evaluations
=============================

The minimum script's summary still has gaps:

  Grid (WER)                 sei test      ncx test
  sei adapter (reference)    MISSING       MISSING (off-diag)
  ncx adapter (reference)    MISSING (off) MISSING
  joint adapter              0.6326        0.6686        ← done
  joint upsampled            —             —             ← needs train first
  third-seed monolinguals    —             —             ← needs train first

Also missing across the board:
  - Morpheme F1 on the joint adapter (the empty table in summary.md)
  - Morpheme F1 on the seed replicates
  - τ-sweeps on monolingual cells (including new third-seed runs)
  - τ-sweep JSONL/tokens for anything that only has a flat predictions CSV

This script fills all of those. It's idempotent — every step checks for
existing output and skips if found. Safe to re-run.

Usage:
  # Fill everything that can currently be filled:
  python run_missing_evaluations.py

  # Only specific phases:
  python run_missing_evaluations.py --only test_metrics    # just the missing cells
  python run_missing_evaluations.py --only off_diagonal    # just off-diagonals
  python run_missing_evaluations.py --only tau_sweeps      # just reconstruction
  python run_missing_evaluations.py --only morpheme_f1     # just morpheme F1
  python run_missing_evaluations.py --only summary         # regenerate summary

  # After running missing training, sweep the new adapters:
  python run_missing_evaluations.py --include_missing_training
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).parent.resolve()

# Reference adapter paths (read-only)
SEI_REF = SCRIPT_DIR / "sei_lora_adapter"
NCX_REF = SCRIPT_DIR / "ncx_lora_adapter"

# Existing experiment outputs (read-only)
MIN_JOINT = SCRIPT_DIR / "min_experiments" / "joint_adapter"
MIN_RESULTS = SCRIPT_DIR / "results" / "minimum"
SEED_RESULTS = SCRIPT_DIR / "results" / "seed_replicates"
SEED_ADAPTERS_LOCAL = SCRIPT_DIR / "experiments" / "seed_replicates"

# New-training outputs (may or may not exist)
MISSING_TRAIN = SCRIPT_DIR / "missing_training"

# Tools already in repo
EVAL_MORPHEME_F1 = SCRIPT_DIR / "eval_morpheme_f1.py"

# New evaluation outputs go here
EVAL_ROOT = SCRIPT_DIR / "missing_evaluations"
EVALS_DIR = EVAL_ROOT / "evals"
SWEEPS_DIR = EVAL_ROOT / "tau_sweeps"
B4_COMPAT_DIR = EVAL_ROOT / "b4_compat"       # CSVs in eval_morpheme_f1 schema
MORPH_DIR = EVAL_ROOT / "morpheme_f1"
SUMMARY_DIR = EVAL_ROOT / "summary"

TAU_VALUES = [0.0, 0.3, 0.5, 0.7, 1.0]
LANGUAGES = ("sei", "ncx")

# --------------------------------------------------------------------------- #
# Safety
# --------------------------------------------------------------------------- #

def assert_safe(p: Path, purpose: str):
    forbidden = [
        SEI_REF, NCX_REF,
        SCRIPT_DIR / "ncx_asr_model" / "ncx-sei-joint-asr",
        MIN_JOINT, MIN_RESULTS, SEED_RESULTS,
        SCRIPT_DIR / "b1_results", SCRIPT_DIR / "b4_results",
    ]
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
# CV data loading
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
    raise FileNotFoundError(f"Cannot find CV data for '{lang}'")


# --------------------------------------------------------------------------- #
# Evaluation with per-word probability extraction
# --------------------------------------------------------------------------- #

def evaluate_adapter_with_tokens(
    adapter_path: Optional[Path],
    language: str,
    output_dir: Path,
    data_dir_override: Optional[str],
    batch_size: int = 16,
    force: bool = False,
) -> Optional[dict]:
    """Evaluate an adapter on language's test set, producing:
        - test_predictions.csv
        - test_tokens.jsonl    (per-word min-p for τ-sweep)
        - test_metrics.json

    Returns the metrics dict or None on failure.
    """
    metrics_file = output_dir / "test_metrics.json"
    tokens_file = output_dir / "test_tokens.jsonl"

    if not force and metrics_file.exists() and tokens_file.exists():
        with open(metrics_file) as f:
            return json.load(f)

    assert_safe(output_dir, f"evaluation {adapter_path} on {language}")
    output_dir.mkdir(parents=True, exist_ok=True)

    import evaluate
    import librosa
    import torch
    import torch.nn.functional as F
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    if adapter_path is not None:
        from peft import PeftModel
        print(f"  loading adapter {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
    else:
        print(f"  (no adapter — zero-shot)")

    model.eval().to(device)

    cv_root = find_cv_root(language, data_dir_override)
    clips_dir = cv_root / "clips"
    examples = []
    with open(cv_root / "test.tsv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            audio_path = clips_dir / row["path"]
            if audio_path.exists():
                sentence = row.get("sentence", row.get("text", "")).lower().strip()
                if sentence:
                    examples.append({"path": str(audio_path), "sentence": sentence})
    print(f"  {len(examples)} test clips for {language}")

    references = [e["sentence"] for e in examples]
    audio_paths = [e["path"] for e in examples]

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
                print(f"  warn {p}: {e}")
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
                        words.append({"word": cur_word.lower().strip(), "min_p": cur_min_p})
                        cur_word = piece.lstrip()
                        cur_min_p = p
                    else:
                        cur_word += piece
                        cur_min_p = min(cur_min_p, p)
                if cur_word.strip():
                    words.append({"word": cur_word.lower().strip(), "min_p": cur_min_p})
                batch_tokens[orig_idx] = words

        predictions.extend(batch_preds)
        token_probs.extend(batch_tokens)
        done = min(i + batch_size, len(audio_paths))
        print(f"  [{done}/{len(audio_paths)}]", end="\r")
    print()

    wer = evaluate.load("wer").compute(predictions=predictions, references=references)
    cer = evaluate.load("cer").compute(predictions=predictions, references=references)

    with open(output_dir / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["language", "reference", "prediction"])
        w.writeheader()
        for ref, pred in zip(references, predictions):
            w.writerow({"language": language, "reference": ref, "prediction": pred})

    with open(tokens_file, "w", encoding="utf-8") as f:
        for ref, pred, probs in zip(references, predictions, token_probs):
            f.write(json.dumps({
                "language": language, "reference": ref,
                "prediction": pred, "words": probs,
            }, ensure_ascii=False) + "\n")

    metrics = {
        "adapter": str(adapter_path) if adapter_path else None,
        "language": language, "wer": wer, "cer": cer, "clips": len(examples),
    }
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  WER {wer:.4f}  CER {cer:.4f}  → {output_dir}")
    return metrics


# --------------------------------------------------------------------------- #
# τ-sweep
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
            curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(ca != cb)))
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


def _simple_wer(preds, refs) -> float:
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


def _simple_cer(preds, refs) -> float:
    return _simple_wer(
        [" ".join(list(p)) for p in preds],
        [" ".join(list(r)) for r in refs],
    )


def build_vocab(tsv_path: Path) -> List[str]:
    vocab = set()
    with open(tsv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            s = row.get("sentence", row.get("text", "")).lower().strip()
            vocab.update(s.split())
    return sorted(vocab)


def run_tau_sweep(
    tokens_jsonl: Path, language: str, vocab: List[str], output_dir: Path,
    tau_values: List[float] = TAU_VALUES, max_edit_dist: int = 2,
    force: bool = False,
) -> Optional[dict]:
    summary_file = output_dir / "threshold_sweep.json"
    if not force and summary_file.exists():
        with open(summary_file) as f:
            return json.load(f)

    assert_safe(output_dir, f"τ-sweep {language}")
    output_dir.mkdir(parents=True, exist_ok=True)

    clips = [json.loads(line) for line in open(tokens_jsonl)]
    references = [c["reference"] for c in clips]
    originals = [c["prediction"] for c in clips]
    wer_base = _simple_wer(originals, references)
    cer_base = _simple_cer(originals, references)

    print(f"  baseline WER {wer_base:.4f} CER {cer_base:.4f}")
    sweep_rows = []
    for tau in tau_values:
        reconstructed = []
        n_rewritten = 0
        for clip in clips:
            out = []
            for w in clip["words"]:
                word, min_p = w["word"], w["min_p"]
                if min_p >= tau:
                    out.append(word)
                else:
                    nw = _closest_word(word, vocab, max_edit_dist)
                    if nw != word: n_rewritten += 1
                    out.append(nw)
            reconstructed.append(" ".join(out))

        wer_after = _simple_wer(reconstructed, references)
        cer_after = _simple_cer(reconstructed, references)
        n_cand = sum(1 for c in clips for w in c["words"] if w["min_p"] < tau)
        n_words = sum(len(c["words"]) for c in clips)
        print(f"  τ={tau:.2f}  WER {wer_after:.4f} (Δ{wer_after-wer_base:+.4f})  "
              f"CER {cer_after:.4f} (Δ{cer_after-cer_base:+.4f})  "
              f"cand {n_cand}/{n_words}  rewr {n_rewritten}")

        csv_path = output_dir / f"recon_tau{tau:.2f}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["language", "reference",
                                              "prediction", "corrected"])
            w.writeheader()
            for ref, orig, corr in zip(references, originals, reconstructed):
                w.writerow({"language": language, "reference": ref,
                            "prediction": orig, "corrected": corr})

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
        "language": language, "tokens_file": str(tokens_jsonl),
        "vocab_size": len(vocab),
        "baseline_wer": wer_base, "baseline_cer": cer_base,
        "max_edit_dist": max_edit_dist, "sweep": sweep_rows,
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# --------------------------------------------------------------------------- #
# Morpheme F1 via the existing eval_morpheme_f1.py
# --------------------------------------------------------------------------- #

def run_morpheme_f1(slug: str, sweep_dir: Path, language: str,
                    output_dir: Path, force: bool = False) -> Optional[dict]:
    result_file = output_dir / f"{slug}_morpheme_f1.json"
    if not force and result_file.exists():
        with open(result_file) as f:
            return json.load(f).get(language)

    assert_safe(output_dir, f"morpheme F1 {slug}")
    output_dir.mkdir(parents=True, exist_ok=True)

    src = sweep_dir / "recon_tau1.00.csv"
    if not src.exists():
        print(f"  [morph-F1] missing {src}, skipping")
        return None

    # eval_morpheme_f1.py expects b4_{lang}_reconstruction.csv in --b4_dir
    staging = B4_COMPAT_DIR / slug
    staging.mkdir(parents=True, exist_ok=True)
    dst = staging / f"b4_{language}_reconstruction.csv"
    shutil.copy2(src, dst)

    if not EVAL_MORPHEME_F1.exists():
        print(f"  [morph-F1] {EVAL_MORPHEME_F1} not found, skipping")
        return None

    tmp_out = staging / "morpheme_f1_results.json"
    result = subprocess.run(
        [sys.executable, str(EVAL_MORPHEME_F1),
         "--lang", language,
         "--b4_dir", str(staging),
         "--output", str(tmp_out)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  [morph-F1] failed:\n{result.stderr[:500]}")
        return None

    if tmp_out.exists():
        with open(tmp_out) as f:
            data = json.load(f)
        shutil.copy2(tmp_out, result_file)
        return data.get(language)
    return None


# --------------------------------------------------------------------------- #
# Gap discovery — which evaluations need to run?
# --------------------------------------------------------------------------- #

def discover_gaps(args) -> List[dict]:
    """Build the list of (slug, adapter, language, reason) that still need evals.

    Each entry is one adapter × test-set pair we need predictions + tokens for.
    """
    gaps = []

    # 1. Reference diagonal: sei adapter on sei (has test_metrics.json but NO tokens)
    gaps.append({
        "slug": "sei_ref_on_sei",
        "adapter": SEI_REF,
        "language": "sei",
        "reason": "Reference sei adapter — need tokens for τ-sweep",
    })

    # 2. Reference diagonal: ncx adapter on ncx (has ADAPTER but NO metrics)
    gaps.append({
        "slug": "ncx_ref_on_ncx",
        "adapter": NCX_REF,
        "language": "ncx",
        "reason": "Reference ncx adapter — missing metrics (the empty tracker cell)",
    })

    # 3. Off-diagonals
    gaps.append({
        "slug": "sei_ref_on_ncx",
        "adapter": SEI_REF,
        "language": "ncx",
        "reason": "Off-diagonal: sei→ncx interference measurement",
    })
    gaps.append({
        "slug": "ncx_ref_on_sei",
        "adapter": NCX_REF,
        "language": "sei",
        "reason": "Off-diagonal: ncx→sei interference measurement",
    })

    # 4. If missing training has run, sweep its outputs too
    if args.include_missing_training and MISSING_TRAIN.exists():
        # Joint upsampled
        joint_up = MISSING_TRAIN / "joint_upsampled" / "adapter"
        if joint_up.exists():
            for lang in LANGUAGES:
                gaps.append({
                    "slug": f"joint_upsampled_on_{lang}",
                    "adapter": joint_up,
                    "language": lang,
                    "reason": f"Joint-upsampled evaluation on {lang}",
                })
        # Third-seed monolinguals (seed=7 default)
        for lang in LANGUAGES:
            for seed_dir in MISSING_TRAIN.glob(f"{lang}_seed*"):
                seed = seed_dir.name.replace(f"{lang}_seed", "")
                adapter = seed_dir / "adapter"
                if adapter.exists():
                    gaps.append({
                        "slug": f"{lang}_seed{seed}_on_{lang}",
                        "adapter": adapter,
                        "language": lang,
                        "reason": f"Third-seed {lang} replicate on {lang} test",
                    })

    # Filter out gaps whose output is already complete (this script is idempotent)
    remaining = []
    for g in gaps:
        existing_metrics = EVALS_DIR / g["slug"] / "test_metrics.json"
        existing_tokens = EVALS_DIR / g["slug"] / "test_tokens.jsonl"
        if existing_metrics.exists() and existing_tokens.exists():
            continue
        remaining.append(g)
    return remaining


# --------------------------------------------------------------------------- #
# Phase: morpheme F1 on existing results (joint, seed replicates)
# --------------------------------------------------------------------------- #

def morpheme_f1_on_existing_results() -> dict:
    """Compute morpheme F1 for every prediction CSV the previous runs produced,
    even though their tau sweeps were never combined with morpheme F1.

    The minimum script's summary.md had an empty morpheme F1 table. This fills it.
    """
    out = {}

    # Joint adapter — we have tau sweep JSONs but no CSV at τ=1.0 in results/minimum/.
    # The sweep CSVs live inside min_experiments/tau_sweeps/<slug>/ on the train machine.
    # Here we check the local sweep results if copied into the repo.
    min_sweeps_local = SCRIPT_DIR / "min_experiments" / "tau_sweeps"
    if min_sweeps_local.exists():
        for slug in ("joint_adapter_on_sei", "joint_adapter_on_ncx"):
            sweep_dir = min_sweeps_local / slug
            lang = slug.split("_")[-1]
            if sweep_dir.exists():
                print(f"\n[morph-F1] {slug}")
                out[slug] = run_morpheme_f1(slug, sweep_dir, lang, MORPH_DIR)
    else:
        # Alternative: if only the JSONL exists under results/minimum/, we cannot
        # produce the τ=1.0 reconstruction CSV without re-reading tokens + vocab.
        # Build it on the fly from the sweep JSON (which preserves the final texts? no,
        # it preserves metrics, not texts). Fall through with a warning.
        for slug in ("joint_adapter_on_sei", "joint_adapter_on_ncx"):
            sweep_json = MIN_RESULTS / f"{slug}_threshold_sweep.json"
            if sweep_json.exists():
                print(f"[morph-F1] {slug}: τ=1.0 CSV missing from repo; "
                      f"skip — run this script on the training machine to "
                      f"regenerate, or include the sweep CSVs.")
                out[slug] = None

    # Seed replicates: we have predictions CSVs (without tokens), so we can
    # compute morpheme F1 on the raw predictions as a baseline-with-no-reconstruction
    # number. This isn't a "B4" column but it's useful for context in the paper.
    seed_pred_dir = SEED_RESULTS / "predictions"
    if seed_pred_dir.exists():
        for csv_path in sorted(seed_pred_dir.glob("*_test_predictions.csv")):
            # Filename pattern: {lang}_seed{N}_test_predictions.csv
            stem = csv_path.stem.replace("_test_predictions", "")
            lang = stem.split("_")[0]
            slug = f"seed_{stem}_baseline"
            staging = B4_COMPAT_DIR / slug
            staging.mkdir(parents=True, exist_ok=True)
            dst = staging / f"b4_{lang}_reconstruction.csv"
            if not dst.exists():
                # Synthesise a "B4 reconstruction" CSV where corrected == prediction
                # (no reconstruction applied). Token F1 then measures baseline
                # morpheme-level quality of the adapter itself.
                with open(csv_path, newline="", encoding="utf-8") as fin:
                    rows = list(csv.DictReader(fin))
                with open(dst, "w", newline="", encoding="utf-8") as fout:
                    w = csv.DictWriter(fout, fieldnames=[
                        "language", "reference", "prediction", "corrected"])
                    w.writeheader()
                    for r in rows:
                        w.writerow({
                            "language": r["language"],
                            "reference": r["reference"],
                            "prediction": r["prediction"],
                            "corrected": r["prediction"],
                        })

            result_file = MORPH_DIR / f"{slug}_morpheme_f1.json"
            if result_file.exists():
                with open(result_file) as f:
                    out[slug] = json.load(f).get(lang)
                continue

            print(f"\n[morph-F1] {slug}")
            tmp_out = staging / "morpheme_f1_results.json"
            MORPH_DIR.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                [sys.executable, str(EVAL_MORPHEME_F1),
                 "--lang", lang, "--b4_dir", str(staging),
                 "--output", str(tmp_out)],
                capture_output=True, text=True,
            )
            if result.returncode == 0 and tmp_out.exists():
                with open(tmp_out) as f:
                    data = json.load(f)
                shutil.copy2(tmp_out, result_file)
                out[slug] = data.get(lang)
            else:
                print(f"  failed: {result.stderr[:300]}")
                out[slug] = None
    return out


# --------------------------------------------------------------------------- #
# Phase orchestration
# --------------------------------------------------------------------------- #

def phase_test_metrics(args, gaps: List[dict]) -> Dict[str, dict]:
    """Produce test_metrics.json + test_tokens.jsonl for every gap."""
    results = {}
    print(f"\n{'='*65}\n  Phase 1: evaluations ({len(gaps)} to run)\n{'='*65}")
    for g in gaps:
        out_dir = EVALS_DIR / g["slug"]
        print(f"\n[eval] {g['slug']} — {g['reason']}")
        if not Path(g["adapter"]).exists():
            print(f"  adapter not found at {g['adapter']}, skip")
            results[g["slug"]] = {"error": "adapter missing"}
            continue
        try:
            m = evaluate_adapter_with_tokens(
                adapter_path=Path(g["adapter"]),
                language=g["language"],
                output_dir=out_dir,
                data_dir_override=args.data_dir,
                batch_size=args.batch_size,
            )
            results[g["slug"]] = m
        except Exception as e:
            print(f"  [eval] FAIL: {e}")
            results[g["slug"]] = {"error": str(e)}
    return results


def phase_tau_sweeps(args, gaps: List[dict]) -> Dict[str, dict]:
    """For every completed eval (new + old), run τ-sweep if not already."""
    print(f"\n{'='*65}\n  Phase 2: τ-sweeps\n{'='*65}")

    # Build vocab once per language
    vocabs = {}
    for lang in LANGUAGES:
        try:
            cv_root = find_cv_root(lang, args.data_dir)
        except FileNotFoundError as e:
            print(f"  vocab[{lang}] skipped: {e}")
            continue
        tsv = cv_root / "train.tsv"
        if not tsv.exists():
            tsv = cv_root / "validated.tsv"
        if tsv.exists():
            vocabs[lang] = build_vocab(tsv)
            print(f"  vocab[{lang}]: {len(vocabs[lang])} words from {tsv.name}")

    sweeps = {}
    # All evaluation outputs under EVALS_DIR
    for g in gaps:
        slug, lang = g["slug"], g["language"]
        tokens = EVALS_DIR / slug / "test_tokens.jsonl"
        if not tokens.exists():
            continue
        if lang not in vocabs:
            continue
        print(f"\n[sweep] {slug}")
        sweeps[slug] = run_tau_sweep(
            tokens_jsonl=tokens, language=lang,
            vocab=vocabs[lang],
            output_dir=SWEEPS_DIR / slug,
        )
    return sweeps


def phase_morpheme_f1(sweeps: Dict[str, dict],
                      gaps_meta: Dict[str, str]) -> Dict[str, dict]:
    """Morpheme F1 on every sweep result (τ=1.0 CSV) and on existing baselines."""
    print(f"\n{'='*65}\n  Phase 3: morpheme F1\n{'='*65}")

    out = {}
    # Morpheme F1 on the new sweeps
    for slug, sw in sweeps.items():
        if not sw:
            continue
        lang = gaps_meta.get(slug, sw.get("language"))
        if lang is None:
            continue
        print(f"\n[morph-F1] {slug}")
        out[slug] = run_morpheme_f1(slug, SWEEPS_DIR / slug, lang, MORPH_DIR)

    # Morpheme F1 on existing joint + seed replicate outputs
    existing = morpheme_f1_on_existing_results()
    out.update(existing)
    return out


# --------------------------------------------------------------------------- #
# Consolidated summary
# --------------------------------------------------------------------------- #

def load_existing_results():
    """Collect everything that's been computed before this script ran."""
    existing = {"min": {}, "seed": {}, "b1": {}, "legacy_b4": {}, "sei_ref": {}}

    # Existing minimum results
    for slug in ("joint_adapter_on_sei", "joint_adapter_on_ncx"):
        mf = MIN_RESULTS / f"{slug}_test_metrics.json"
        sf = MIN_RESULTS / f"{slug}_threshold_sweep.json"
        if mf.exists():
            with open(mf) as f: existing["min"][f"{slug}_metrics"] = json.load(f)
        if sf.exists():
            with open(sf) as f: existing["min"][f"{slug}_sweep"] = json.load(f)

    # Existing seed replicate metrics
    metrics_dir = SEED_RESULTS / "metrics"
    if metrics_dir.exists():
        for f in metrics_dir.glob("*.json"):
            with open(f) as fh:
                existing["seed"][f.stem] = json.load(fh)

    # B1 baselines
    b1_json = SCRIPT_DIR / "b1_results" / "b1_zero_shot_metrics.json"
    if b1_json.exists():
        with open(b1_json) as f:
            existing["b1"] = json.load(f)

    # Legacy B4 (reconstruction on B1 predictions)
    lb4 = SCRIPT_DIR / "b4_results" / "morpheme_f1_results.json"
    if lb4.exists():
        with open(lb4) as f:
            existing["legacy_b4"] = json.load(f)

    # Reference sei adapter (the test_metrics.json that already exists)
    sei_ref_metrics = SEI_REF / "test_metrics.json"
    if sei_ref_metrics.exists():
        with open(sei_ref_metrics) as f:
            existing["sei_ref"] = json.load(f)

    return existing


def emit_summary(args, new_eval: dict, new_sweep: dict, new_morph: dict):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_existing_results()

    # ---------- Consolidated WER grid ----------
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    def std(xs):
        xs = [x for x in xs if x is not None]
        if len(xs) < 2:
            return 0.0
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

    # Monolingual WER per language = (seed42 + seed13 + new_seed7? + ref)
    def monolingual_wer(lang):
        wers = []
        # Seed replicates
        for seed in (42, 13):
            k = f"{lang}_seed{seed}_test_metrics"
            if k in existing["seed"]:
                wers.append(existing["seed"][k].get("test_wer"))
        # Third seed if run
        for slug, res in new_eval.items():
            if slug.startswith(f"{lang}_seed") and slug.endswith(f"on_{lang}"):
                if isinstance(res, dict) and "wer" in res:
                    wers.append(res["wer"])
        # Reference adapter re-eval (if matches the language)
        ref_slug = f"{'sei' if lang=='sei' else 'ncx'}_ref_on_{lang}"
        if ref_slug in new_eval and isinstance(new_eval[ref_slug], dict):
            w = new_eval[ref_slug].get("wer")
            if w is not None:
                wers.append(w)
        return wers

    sei_wers = monolingual_wer("sei")
    ncx_wers = monolingual_wer("ncx")

    # Off-diagonals
    sei_on_ncx = new_eval.get("sei_ref_on_ncx", {}).get("wer")
    ncx_on_sei = new_eval.get("ncx_ref_on_sei", {}).get("wer")

    # Joint (unbalanced)
    joint_on_sei = existing["min"].get("joint_adapter_on_sei_metrics", {}).get("wer")
    joint_on_ncx = existing["min"].get("joint_adapter_on_ncx_metrics", {}).get("wer")

    # Joint upsampled
    joint_up_sei = new_eval.get("joint_upsampled_on_sei", {}).get("wer")
    joint_up_ncx = new_eval.get("joint_upsampled_on_ncx", {}).get("wer")

    md = ["# Shout — consolidated results after missing-evaluations run\n"]

    md.append("## WER grid\n")
    md.append("| Configuration | sei test | ncx test |")
    md.append("|---|---|---|")
    md.append("| Zero-shot Whisper-small (B1) | "
              f"{next((r['wer'] for r in existing['b1'] if r['language']=='sei'), '—'):.4f} | "
              f"{next((r['wer'] for r in existing['b1'] if r['language']=='ncx'), '—'):.4f} |")
    fmt = lambda xs: (f"{mean(xs):.4f} ± {std(xs):.4f} (n={len([x for x in xs if x])})"
                      if mean(xs) is not None else "—")
    md.append(f"| sei adapter (monolingual) | {fmt(sei_wers)} | "
              f"{f'{sei_on_ncx:.4f}' if sei_on_ncx else '—'} |")
    md.append(f"| ncx adapter (monolingual) | "
              f"{f'{ncx_on_sei:.4f}' if ncx_on_sei else '—'} | {fmt(ncx_wers)} |")
    md.append(f"| joint adapter (unbalanced) | "
              f"{f'{joint_on_sei:.4f}' if joint_on_sei else '—'} | "
              f"{f'{joint_on_ncx:.4f}' if joint_on_ncx else '—'} |")
    if joint_up_sei or joint_up_ncx:
        md.append(f"| joint adapter (upsampled) | "
                  f"{f'{joint_up_sei:.4f}' if joint_up_sei else '—'} | "
                  f"{f'{joint_up_ncx:.4f}' if joint_up_ncx else '—'} |")
    md.append("")

    # ---------- τ-sweep table ----------
    md.append("## τ-sweep (WER after reconstruction)\n")
    md.append("| Configuration | τ=0.0 | τ=0.3 | τ=0.5 | τ=0.7 | τ=1.0 |")
    md.append("|---|---|---|---|---|---|")

    def sweep_row(label, sweep_data):
        if not sweep_data or "sweep" not in sweep_data:
            return None
        cells = [f"**{label}**"]
        for s in sweep_data["sweep"]:
            cells.append(f"{s['wer_after']:.4f}")
        return "| " + " | ".join(cells) + " |"

    for slug in ("joint_adapter_on_sei", "joint_adapter_on_ncx"):
        sd = existing["min"].get(f"{slug}_sweep")
        row = sweep_row(slug, sd)
        if row: md.append(row)
    for slug, sd in new_sweep.items():
        row = sweep_row(slug, sd)
        if row: md.append(row)
    md.append("")

    # ---------- Morpheme F1 table ----------
    md.append("## Morpheme F1 (word-level token F1 proxy)\n")
    md.append("| Configuration | Language | Token F1 before | Token F1 after | Δ |")
    md.append("|---|---|---|---|---|")

    # Legacy B4-on-B1
    for lang, d in existing["legacy_b4"].items():
        md.append(f"| legacy B4 (B1 preds + recon) | {lang} | "
                  f"{d['token_f1_before']:.4f} | {d['token_f1_after']:.4f} | "
                  f"{d['token_f1_delta']:+.4f} |")

    for slug, mf in new_morph.items():
        if not mf: continue
        lang = mf.get("language", "?")
        md.append(f"| {slug} | {lang} | {mf.get('token_f1_before', 0):.4f} | "
                  f"{mf.get('token_f1_after', 0):.4f} | "
                  f"{mf.get('token_f1_delta', 0):+.4f} |")
    md.append("")

    # ---------- Interpretation hooks ----------
    md.append("## Notes for the paper\n")
    md.append("- Monolingual WERs are reported as mean ± std across all available seeds "
              "(including the original reference adapter if re-evaluated via this script).")
    md.append("- Joint (unbalanced) uses plain concatenation, matching the archived XLS-R "
              "methodology.")
    md.append("- Joint (upsampled) repeats ncx entries to match sei volume. Comparing the "
              "two joint rows disambiguates capacity interference from data dilution.")
    md.append("- τ=1.0 is the classic B4 always-on; τ<1.0 is B4.5 (uncertainty-triggered). "
              "Best τ is typically 0.5–0.7 in these runs.")
    md.append("- Morpheme F1 is computed via `eval_morpheme_f1.py`. The metric is "
              "word-level token F1 and word-boundary F1 — an ad-hoc proxy, since no "
              "morphological analyser exists for sei/ncx.")

    (SUMMARY_DIR / "consolidated_summary.md").write_text("\n".join(md), encoding="utf-8")

    # JSON dump
    with open(SUMMARY_DIR / "consolidated_summary.json", "w") as f:
        json.dump({
            "new_evaluations": new_eval,
            "new_tau_sweeps": new_sweep,
            "new_morpheme_f1": new_morph,
            "existing": existing,
        }, f, indent=2, default=str)

    # Flat CSV of every (adapter, language) × τ with WER/CER
    csv_rows = []
    # Joint unbalanced
    for slug in ("joint_adapter_on_sei", "joint_adapter_on_ncx"):
        sd = existing["min"].get(f"{slug}_sweep")
        if sd:
            for s in sd["sweep"]:
                csv_rows.append({
                    "config": slug, "language": sd["language"],
                    "tau": s["threshold"],
                    "wer": s["wer_after"], "cer": s["cer_after"],
                    "wer_before": s["wer_before"], "cer_before": s["cer_before"],
                })
    for slug, sd in new_sweep.items():
        if not sd: continue
        for s in sd["sweep"]:
            csv_rows.append({
                "config": slug, "language": sd["language"],
                "tau": s["threshold"],
                "wer": s["wer_after"], "cer": s["cer_after"],
                "wer_before": s["wer_before"], "cer_before": s["cer_before"],
            })

    with open(SUMMARY_DIR / "consolidated_results.csv", "w", newline="", encoding="utf-8") as f:
        if csv_rows:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)

    print(f"\n{'='*65}")
    print(f"Summary written:")
    print(f"  {SUMMARY_DIR / 'consolidated_summary.md'}")
    print(f"  {SUMMARY_DIR / 'consolidated_summary.json'}")
    print(f"  {SUMMARY_DIR / 'consolidated_results.csv'}")
    print(f"{'='*65}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Shout — fill missing evaluations")
    parser.add_argument("--only", choices=["test_metrics", "off_diagonal",
                                            "tau_sweeps", "morpheme_f1",
                                            "summary", "all"],
                        default="all")
    parser.add_argument("--include_missing_training", action="store_true",
                        help="Also evaluate joint_upsampled + third_seed adapters if present")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--force_reeval", action="store_true",
                        help="Re-evaluate even if existing metrics + tokens found")
    args = parser.parse_args()

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"{'='*65}")
    print(f"  Shout — missing evaluations")
    print(f"  Phase: {args.only}")
    print(f"  Output root: {EVAL_ROOT}")
    print(f"  Include new-training adapters: {args.include_missing_training}")
    print(f"{'='*65}")

    gaps = discover_gaps(args)
    print(f"\nGaps discovered:")
    for g in gaps:
        print(f"  {g['slug']:<35} — {g['reason']}")
    if not gaps:
        print("  (nothing to evaluate)")
    gaps_meta = {g["slug"]: g["language"] for g in gaps}

    new_eval, new_sweep, new_morph = {}, {}, {}

    if args.only in ("test_metrics", "off_diagonal", "all"):
        # "off_diagonal" is a subset of test_metrics; filter if requested
        run_gaps = gaps
        if args.only == "off_diagonal":
            run_gaps = [g for g in gaps if "on" in g["slug"] and
                        g["slug"].split("_on_")[0].startswith(
                            ("sei_ref", "ncx_ref")) and
                        g["language"] != (g["slug"].split("_")[0])]
        new_eval = phase_test_metrics(args, run_gaps)

    if args.only in ("tau_sweeps", "all"):
        # Need completed evals — either just produced or on disk
        eval_slugs_with_tokens = []
        for g in gaps:
            if (EVALS_DIR / g["slug"] / "test_tokens.jsonl").exists():
                eval_slugs_with_tokens.append(g)
        new_sweep = phase_tau_sweeps(args, eval_slugs_with_tokens)

    if args.only in ("morpheme_f1", "all"):
        new_morph = phase_morpheme_f1(new_sweep, gaps_meta)

    if args.only in ("summary", "all"):
        # Reload anything produced but not in memory
        if not new_eval:
            for g in gaps:
                mf = EVALS_DIR / g["slug"] / "test_metrics.json"
                if mf.exists():
                    with open(mf) as f: new_eval[g["slug"]] = json.load(f)
        if not new_sweep:
            for g in gaps:
                sf = SWEEPS_DIR / g["slug"] / "threshold_sweep.json"
                if sf.exists():
                    with open(sf) as f: new_sweep[g["slug"]] = json.load(f)
        if not new_morph:
            for f in MORPH_DIR.glob("*_morpheme_f1.json") if MORPH_DIR.exists() else []:
                with open(f) as fh:
                    slug = f.stem.replace("_morpheme_f1", "")
                    data = json.load(fh)
                    for lang in data:
                        new_morph[slug] = data[lang]
        emit_summary(args, new_eval, new_sweep, new_morph)

    print("\n✓ Done.")


if __name__ == "__main__":
    main()