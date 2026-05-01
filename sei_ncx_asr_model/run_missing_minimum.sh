#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
EVAL="$ROOT/experiments/seed_replicates/eval_adapter.py"

# Change these if your adapters are elsewhere
SEI_ADAPTER="${SEI_ADAPTER:-$ROOT/sei_lora_adapter}"
NCX_ADAPTER="${NCX_ADAPTER:-$ROOT/ncx_lora_adapter}"

OUT="$ROOT/results/minimum"
mkdir -p "$OUT"

echo "Using:"
echo "  ROOT=$ROOT"
echo "  SEI_ADAPTER=$SEI_ADAPTER"
echo "  NCX_ADAPTER=$NCX_ADAPTER"
echo

run_eval () {
  local adapter="$1"
  local language="$2"
  local outdir="$3"

  echo "=== Evaluating adapter=$adapter on language=$language -> $outdir ==="
  python "$EVAL" \
    --adapter "$adapter" \
    --language "$language" \
    --output_dir "$outdir"
  echo
}

# Diagonal
run_eval "$SEI_ADAPTER" "sei" "$OUT/sei_adapter_on_sei"
run_eval "$NCX_ADAPTER" "ncx" "$OUT/ncx_adapter_on_ncx"

# Off-diagonal
run_eval "$SEI_ADAPTER" "ncx" "$OUT/sei_adapter_on_ncx"
run_eval "$NCX_ADAPTER" "sei" "$OUT/ncx_adapter_on_sei"

echo "Done. Generated:"
find "$OUT" -maxdepth 2 -type f \( -name "test_metrics.json" -o -name "test_predictions.csv" -o -name "test_tokens.jsonl" \) | sort