#!/usr/bin/env python3
"""
Shout — Aggregate all experiment results
==========================================

Walks experiments/ and produces:
  - summary_all.csv              — one row per run with config + metrics
  - summary_by_config.csv        — mean ± std over seeds (when seeds exist)
  - aggregate.md                 — markdown tables suitable for the paper draft

Uses the test_metrics.json + run_config.json that train_whisper_lora.py writes.

Run this after any phase of run_all_experiments.sh completes. Safe to re-run
at any time — it's read-only over the experiments/ directory.
"""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from collections import defaultdict


def collect_runs(root: Path):
    """Walk root/*, collect (config, metrics) for every run that has both."""
    rows = []
    for run_dir in root.rglob("run_config.json"):
        run_dir = run_dir.parent
        metrics_path = run_dir / "test_metrics.json"
        if not metrics_path.exists():
            continue
        with open(run_dir / "run_config.json") as f:
            config = json.load(f)
        with open(metrics_path) as f:
            metrics = json.load(f)
        # B4.5 threshold sweep results (separate format)
        row = {
            "run_dir": str(run_dir.relative_to(root)),
            "language": config.get("language"),
            "base_model": config.get("base_model", "small"),
            "lora_rank": config.get("lora_rank"),
            "lora_target": config.get("lora_target"),
            "seed": config.get("seed"),
            "data_fraction": config.get("data_fraction", 1.0),
            "wer": metrics.get("test_wer") or metrics.get("wer"),
            "cer": metrics.get("test_cer") or metrics.get("cer"),
        }
        rows.append(row)
    return rows


def group_by_config(rows):
    """Group rows by (language, base_model, rank, target, fraction), aggregating over seeds."""
    groups = defaultdict(list)
    for r in rows:
        key = (r["language"], r["base_model"], r["lora_rank"],
               r["lora_target"], r["data_fraction"])
        groups[key].append(r)

    aggregated = []
    for key, group_rows in sorted(groups.items()):
        wers = [r["wer"] for r in group_rows if r["wer"] is not None]
        cers = [r["cer"] for r in group_rows if r["cer"] is not None]
        seeds = sorted(set(r["seed"] for r in group_rows if r["seed"] is not None))
        agg = {
            "language": key[0],
            "base_model": key[1],
            "lora_rank": key[2],
            "lora_target": key[3],
            "data_fraction": key[4],
            "n_seeds": len(seeds),
            "seeds": ",".join(str(s) for s in seeds),
            "wer_mean": statistics.mean(wers) if wers else None,
            "wer_std": statistics.stdev(wers) if len(wers) > 1 else 0.0,
            "cer_mean": statistics.mean(cers) if cers else None,
            "cer_std": statistics.stdev(cers) if len(cers) > 1 else 0.0,
        }
        aggregated.append(agg)
    return aggregated


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def format_md_table(rows, cols):
    if not rows:
        return "(no rows)"
    headers = [c[0] for c in cols]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        cells = []
        for _, key, fmt in cols:
            v = r.get(key)
            if v is None:
                cells.append("—")
            elif fmt:
                cells.append(fmt.format(v) if not isinstance(v, str) else v)
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def collect_b45_sweep(root: Path):
    """Collect B4.5 threshold sweep results."""
    results = []
    for summary_path in root.rglob("threshold_sweep.json"):
        with open(summary_path) as f:
            summary = json.load(f)
        for sweep in summary["sweep"]:
            results.append({
                "language": summary["language"],
                "threshold": sweep["threshold"],
                "wer_before": sweep["wer_before"],
                "wer_after": sweep["wer_after"],
                "wer_delta": sweep["wer_delta"],
                "n_rewritten": sweep["n_rewritten"],
                "candidates_pct": 100 * sweep["n_candidates"] / max(1, sweep["n_words"]),
            })
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="./experiments",
                   help="Experiments directory to walk")
    p.add_argument("--output_dir", default="./experiments/_summary")
    args = p.parse_args()

    root = Path(args.root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Main aggregation
    rows = collect_runs(root)
    print(f"Collected {len(rows)} individual runs")
    write_csv(out_dir / "summary_all.csv", rows,
              fieldnames=["run_dir", "language", "base_model", "lora_rank",
                          "lora_target", "seed", "data_fraction", "wer", "cer"])

    aggregated = group_by_config(rows)
    print(f"Grouped into {len(aggregated)} configurations")
    write_csv(out_dir / "summary_by_config.csv", aggregated,
              fieldnames=["language", "base_model", "lora_rank", "lora_target",
                          "data_fraction", "n_seeds", "seeds",
                          "wer_mean", "wer_std", "cer_mean", "cer_std"])

    # B4.5 sweep
    b45 = collect_b45_sweep(root)
    if b45:
        write_csv(out_dir / "summary_b45_sweep.csv", b45,
                  fieldnames=["language", "threshold", "wer_before", "wer_after",
                              "wer_delta", "n_rewritten", "candidates_pct"])

    # Markdown summary
    md = ["# Shout experiment summary\n"]

    # Filter aggregations by kind of experiment for separate tables
    def filter_rows(pred):
        return [r for r in aggregated if pred(r)]

    seed_reps = filter_rows(lambda r: r["n_seeds"] >= 2
                                       and r["lora_rank"] == 32
                                       and r["lora_target"] == "both"
                                       and r["data_fraction"] == 1.0
                                       and r["base_model"] == "small")
    if seed_reps:
        md.append("## Seed replicates (rank=32, target=both, small, full data)\n")
        md.append(format_md_table(seed_reps, [
            ("Lang", "language", None),
            ("Seeds", "seeds", None),
            ("WER (mean ± std)", "wer_mean", "{:.4f}"),
            ("WER std", "wer_std", "{:.4f}"),
            ("CER (mean)", "cer_mean", "{:.4f}"),
        ]))
        md.append("")

    rank_ablation = filter_rows(lambda r: r["lora_target"] == "both"
                                           and r["data_fraction"] == 1.0
                                           and r["base_model"] == "small")
    rank_ablation = [r for r in rank_ablation if r["lora_rank"] in [4, 8, 16, 32, 64]]
    if rank_ablation:
        md.append("## LoRA rank ablation\n")
        md.append(format_md_table(rank_ablation, [
            ("Lang", "language", None),
            ("Rank", "lora_rank", None),
            ("WER", "wer_mean", "{:.4f}"),
            ("CER", "cer_mean", "{:.4f}"),
        ]))
        md.append("")

    placement = filter_rows(lambda r: r["lora_target"] in ["encoder", "decoder", "both"]
                                       and r["lora_rank"] == 32
                                       and r["data_fraction"] == 1.0
                                       and r["base_model"] == "small")
    if placement:
        md.append("## LoRA placement ablation\n")
        md.append(format_md_table(placement, [
            ("Lang", "language", None),
            ("Placement", "lora_target", None),
            ("WER", "wer_mean", "{:.4f}"),
            ("CER", "cer_mean", "{:.4f}"),
        ]))
        md.append("")

    curve = filter_rows(lambda r: r["data_fraction"] < 1.0 or
                                   (r["lora_rank"] == 32 and r["lora_target"] == "both"
                                    and r["base_model"] == "small"))
    curve = [r for r in curve if r["base_model"] == "small" and r["lora_target"] == "both"]
    if curve:
        md.append("## Data-subsampling learning curve\n")
        md.append(format_md_table(
            sorted(curve, key=lambda r: (r["language"], r["data_fraction"])),
            [
                ("Lang", "language", None),
                ("Fraction", "data_fraction", "{:.2f}"),
                ("WER", "wer_mean", "{:.4f}"),
                ("CER", "cer_mean", "{:.4f}"),
            ]
        ))
        md.append("")

    size = filter_rows(lambda r: r["base_model"] in ["tiny", "base", "medium"]
                                  and r["lora_rank"] == 32
                                  and r["lora_target"] == "both"
                                  and r["data_fraction"] == 1.0)
    if size:
        md.append("## Model-size sweep\n")
        md.append(format_md_table(
            sorted(size, key=lambda r: (r["language"], r["base_model"])),
            [
                ("Lang", "language", None),
                ("Base model", "base_model", None),
                ("WER", "wer_mean", "{:.4f}"),
                ("CER", "cer_mean", "{:.4f}"),
            ]
        ))
        md.append("")

    if b45:
        md.append("## B4.5 threshold sweep\n")
        md.append(format_md_table(
            sorted(b45, key=lambda r: (r["language"], r["threshold"])),
            [
                ("Lang", "language", None),
                ("τ", "threshold", "{:.2f}"),
                ("WER before", "wer_before", "{:.4f}"),
                ("WER after", "wer_after", "{:.4f}"),
                ("Δ", "wer_delta", "{:+.4f}"),
                ("Candidates %", "candidates_pct", "{:.1f}"),
            ]
        ))

    md_path = out_dir / "aggregate.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\n→ {out_dir}/summary_all.csv       ({len(rows)} runs)")
    print(f"→ {out_dir}/summary_by_config.csv ({len(aggregated)} configs)")
    if b45:
        print(f"→ {out_dir}/summary_b45_sweep.csv ({len(b45)} threshold rows)")
    print(f"→ {out_dir}/aggregate.md")


if __name__ == "__main__":
    main()
