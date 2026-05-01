#!/usr/bin/env python3
import json
import csv
from pathlib import Path

ROOT = Path(".").resolve()
OUT = ROOT / "results" / "minimum"

CONFIGS = [
    ("sei_adapter_on_sei", "B2 diagonal (reference)", "sei"),
    ("ncx_adapter_on_ncx", "B2 diagonal", "ncx"),
    ("joint_adapter_on_sei", "B3 on sei", "sei"),
    ("joint_adapter_on_ncx", "B3 on ncx", "ncx"),
    ("sei_adapter_on_ncx", "off-diagonal: sei→ncx", "ncx"),
    ("ncx_adapter_on_sei", "off-diagonal: ncx→sei", "sei"),
]

TAUS = [0.00, 0.30, 0.50, 0.70, 1.00]

def read_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def fmt(x, digits=4):
    if x is None:
        return ""
    return f"{x:.{digits}f}"

eval_results = {}
tau_sweeps = {}

# Existing joint files may already be flat files in results/minimum/
for slug, _, _ in CONFIGS:
    flat_metrics = OUT / f"{slug}_test_metrics.json"
    nested_metrics = OUT / slug / "test_metrics.json"
    metrics = read_json(flat_metrics) or read_json(nested_metrics)
    if metrics:
        eval_results[slug] = metrics

    flat_tau = OUT / f"{slug}_threshold_sweep.json"
    nested_tau = OUT / slug / "threshold_sweep.json"
    tau = read_json(flat_tau) or read_json(nested_tau)
    if tau:
        tau_sweeps[slug] = tau

# summary.csv
csv_path = OUT / "summary.csv"
fieldnames = [
    "slug", "description", "language", "B2_wer", "B2_cer", "clips",
    "tau0.00_wer", "tau0.00_delta",
    "tau0.30_wer", "tau0.30_delta",
    "tau0.50_wer", "tau0.50_delta",
    "tau0.70_wer", "tau0.70_delta",
    "tau1.00_wer", "tau1.00_delta",
]

rows = []
for slug, desc, lang in CONFIGS:
    m = eval_results.get(slug, {})
    row = {
        "slug": slug,
        "description": desc,
        "language": lang,
        "B2_wer": fmt(m.get("wer")),
        "B2_cer": fmt(m.get("cer")),
        "clips": m.get("clips", ""),
        "tau0.00_wer": "", "tau0.00_delta": "",
        "tau0.30_wer": "", "tau0.30_delta": "",
        "tau0.50_wer": "", "tau0.50_delta": "",
        "tau0.70_wer": "", "tau0.70_delta": "",
        "tau1.00_wer": "", "tau1.00_delta": "",
    }

    sweep = tau_sweeps.get(slug, {}).get("sweep", [])
    for item in sweep:
        tau = item.get("threshold")
        key = f"tau{tau:.2f}"
        row[f"{key}_wer"] = fmt(item.get("wer_after"))
        row[f"{key}_delta"] = fmt(item.get("wer_delta"))

    rows.append(row)

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# summary.json
summary_json = {
    "eval_results": eval_results,
    "tau_sweeps": tau_sweeps,
    "morpheme_f1": {}
}
with open(OUT / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_json, f, indent=2, ensure_ascii=False)

# summary.md
def metric_cell(slug):
    m = eval_results.get(slug)
    if not m:
        return "—"
    return f"{m.get('wer', 0):.4f} / {m.get('cer', 0):.4f}"

md = []
md.append("# Shout — minimum experiment suite results\n")

md.append("## Results grid (WER / CER)\n")
md.append("| Configuration | sei test | ncx test |")
md.append("|---|---|---|")
md.append(f"| sei adapter | {metric_cell('sei_adapter_on_sei')} | {metric_cell('sei_adapter_on_ncx')} |")
md.append(f"| ncx adapter | {metric_cell('ncx_adapter_on_sei')} | {metric_cell('ncx_adapter_on_ncx')} |")
md.append(f"| joint adapter | {metric_cell('joint_adapter_on_sei')} | {metric_cell('joint_adapter_on_ncx')} |")
md.append("")

md.append("## τ-sweep (WER only)\n")
md.append("| Config | τ=0.0 | τ=0.3 | τ=0.5 | τ=0.7 | τ=1.0 |")
md.append("|---|---|---|---|---|---|")
for slug in [
    "sei_adapter_on_sei",
    "ncx_adapter_on_ncx",
    "joint_adapter_on_sei",
    "joint_adapter_on_ncx",
    "sei_adapter_on_ncx",
    "ncx_adapter_on_sei",
]:
    sweep = tau_sweeps.get(slug, {}).get("sweep", [])
    values = {f"{x['threshold']:.2f}": f"{x['wer_after']:.4f}" for x in sweep}
    label = slug
    md.append(
        f"| {label} | "
        f"{values.get('0.00', '—')} | "
        f"{values.get('0.30', '—')} | "
        f"{values.get('0.50', '—')} | "
        f"{values.get('0.70', '—')} | "
        f"{values.get('1.00', '—')} |"
    )

with open(OUT / "summary.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")

print(f"Wrote:\n- {csv_path}\n- {OUT / 'summary.json'}\n- {OUT / 'summary.md'}")