#!/usr/bin/env bash
# =============================================================================
# Shout — Full experiment suite on 4x A100
# =============================================================================
#
# Runs all Tier 1 (paper-critical) and Tier 2 (strong section) experiments.
# Tier 3 is deliberately excluded (see the plan for reasoning).
#
# Directory layout produced:
#   experiments/
#     seed_replicates/        # Tier 1a  — 3 seeds × {sei, ncx, joint}
#       sei_seed{42,13,7}/
#       ncx_seed{42,13,7}/
#       joint_seed{42,13,7}/
#     rank_ablation/          # Tier 1b  — 5 ranks × {sei, ncx}
#       {sei,ncx}_r{4,8,16,32,64}/
#     placement/              # Tier 1c  — 3 placements × {sei, ncx}
#       {sei,ncx}_{encoder,decoder,both}/
#     b45_sweep/              # Tier 1d  — threshold sweep (per-language)
#       {sei,ncx}/
#     data_curve/             # Tier 2a  — 5 sizes × {sei, ncx}
#       {sei,ncx}_frac{0.02,0.05,0.1,0.25,0.5,1.0}/
#     size_sweep/             # Tier 2b  — {base, small, medium} × {sei, ncx}
#       {sei,ncx}_{base,small,medium}/
#     joint_upsampled/        # Tier 2c  — joint with ncx upsampled to 1:1
#       joint_upsample/
#
# Nothing in this script writes to:
#   ./sei_lora_adapter/ or ./ncx_lora_adapter/        (paper's reference B2)
#   ./ncx_asr_model/ncx-sei-joint-asr/                 (paper's joint baseline)
#   ./b1_results/  ./b4_results/                       (paper's reference CSVs)
#
# All new outputs go under ./experiments/.
#
# Usage:
#   bash run_all_experiments.sh [tier1 | tier2 | all | <phase_name>]
#
# Examples:
#   bash run_all_experiments.sh tier1              # ~52 GPU-hours, ~13h wall
#   bash run_all_experiments.sh seed_replicates    # just Tier 1a
#   bash run_all_experiments.sh rank_ablation      # just Tier 1b
#   bash run_all_experiments.sh all                # everything (~150 GPU-h)
#
# =============================================================================

set -euo pipefail

# Resolve script paths
SHOUT_ROOT="${SHOUT_ROOT:-$(pwd)}"
TRAIN="$SHOUT_ROOT/experiments/train_whisper_lora.py"
EVAL="$SHOUT_ROOT/experiments/eval_adapter.py"
B45_SWEEP="$SHOUT_ROOT/experiments/reconstruct_threshold_sweep.py"
OUT="$SHOUT_ROOT/experiments"

mkdir -p "$OUT"

# Expected number of GPUs
NGPU="${NGPU:-4}"

# Utility — run a job on a specific GPU
# Usage: gpu_run <gpu_id> <logfile> <cmd...>
gpu_run() {
  local gpu="$1"; shift
  local log="$1"; shift
  mkdir -p "$(dirname "$log")"
  echo "[gpu $gpu] $* > $log"
  CUDA_VISIBLE_DEVICES="$gpu" "$@" > "$log" 2>&1
}
export -f gpu_run

# Pool executor — run N jobs at a time, one per GPU
# Accepts a series of commands on stdin; each line is run on the next available GPU.
run_pool() {
  local njobs=0
  while IFS= read -r cmd; do
    [ -z "$cmd" ] && continue
    local gpu=$((njobs % NGPU))
    eval "CUDA_VISIBLE_DEVICES=$gpu $cmd" &
    njobs=$((njobs + 1))
    # When the pool is full, wait for one to finish before continuing
    if (( njobs % NGPU == 0 )); then
      wait -n
    fi
  done
  wait   # drain remaining jobs
}

# =============================================================================
# Tier 1a: Seed replicates
# =============================================================================
tier1a_seed_replicates() {
  echo "=== Tier 1a: seed replicates (9 runs) ==="
  local dir="$OUT/seed_replicates"
  mkdir -p "$dir"

  # 3 seeds × 3 configurations. Joint is run via --language both
  # (we're sharing the train script; joint handling is explicit below).
  {
    for seed in 42 13 7; do
      for lang in sei ncx; do
        local run_dir="$dir/${lang}_seed${seed}"
        [ -d "$run_dir/adapter" ] && { echo "# skip existing: $run_dir"; continue; }
        echo "python $TRAIN --language $lang --seed $seed \
              --output_dir $run_dir \
              --wandb_run_name ${lang}-seed${seed} \
              > $run_dir.log 2>&1"
      done
    done
  } | run_pool

  # Joint seeds run separately (one per seed since joint data ~= 5× single-lang)
  # Joint requires a special --language flag we don't expose; use a small wrapper
  # or, if that flag isn't there, comment this block and use the frozen joint
  # model for all three "joint" slots (no seed variance, documented as such).
  echo ""
  echo "NOTE: Joint-model seed replicates require a joint-data loader that the"
  echo "current train_whisper_lora.py does NOT implement — it trains per-language."
  echo "For joint replicates, either:"
  echo "  (a) Add joint data loading by concatenating sei+ncx CV tsv files, OR"
  echo "  (b) Report joint with a single seed and explicitly note the caveat in the paper."
  echo "Skipping joint replicates in this script; document as limitation."
  echo ""
}

# =============================================================================
# Tier 1b: LoRA rank ablation
# =============================================================================
tier1b_rank_ablation() {
  echo "=== Tier 1b: LoRA rank ablation (10 runs) ==="
  local dir="$OUT/rank_ablation"
  mkdir -p "$dir"

  # 5 ranks × 2 languages, all at seed=42
  {
    for rank in 4 8 16 32 64; do
      for lang in sei ncx; do
        local run_dir="$dir/${lang}_r${rank}"
        [ -d "$run_dir/adapter" ] && { echo "# skip existing: $run_dir"; continue; }
        echo "python $TRAIN --language $lang --lora_rank $rank \
              --output_dir $run_dir \
              --wandb_run_name ${lang}-r${rank} \
              > $run_dir.log 2>&1"
      done
    done
  } | run_pool
}

# =============================================================================
# Tier 1c: Encoder vs decoder vs both placement
# =============================================================================
tier1c_placement() {
  echo "=== Tier 1c: LoRA placement ablation (6 runs) ==="
  local dir="$OUT/placement"
  mkdir -p "$dir"

  # 3 placements × 2 languages.
  # "both" is nominally the same as the frozen baseline at rank=32 seed=42,
  # but we run it here again at the same rank for a fair comparison
  # against encoder-only and decoder-only variants.
  {
    for placement in encoder decoder both; do
      for lang in sei ncx; do
        local run_dir="$dir/${lang}_${placement}"
        [ -d "$run_dir/adapter" ] && { echo "# skip existing: $run_dir"; continue; }
        echo "python $TRAIN --language $lang --lora_target $placement \
              --output_dir $run_dir \
              --wandb_run_name ${lang}-${placement} \
              > $run_dir.log 2>&1"
      done
    done
  } | run_pool
}

# =============================================================================
# Tier 1d: B4.5 threshold sweep
# =============================================================================
tier1d_b45_sweep() {
  echo "=== Tier 1d: B4.5 threshold sweep ==="
  local dir="$OUT/b45_sweep"
  mkdir -p "$dir"

  # Precondition: we need per-word probabilities from a trained adapter.
  # Use the seed=42 replicate from Tier 1a (it matches the paper baseline).
  # If Tier 1a hasn't run, use the frozen reference adapters.

  for lang in sei ncx; do
    local tokens_file
    if [ -f "$OUT/seed_replicates/${lang}_seed42/test_tokens.jsonl" ]; then
      tokens_file="$OUT/seed_replicates/${lang}_seed42/test_tokens.jsonl"
    else
      # Generate token probabilities from the reference adapter
      echo "  Generating token probs for $lang from reference adapter..."
      CUDA_VISIBLE_DEVICES=0 python "$EVAL" \
        --adapter "$SHOUT_ROOT/${lang}_lora_adapter" \
        --language "$lang" \
        --output_dir "$dir/${lang}_source" \
        > "$dir/${lang}_source.log" 2>&1
      tokens_file="$dir/${lang}_source/test_tokens.jsonl"
    fi

    # Locate the training TSV (for vocab)
    # This path needs to match the data layout on the A100 machine.
    # Adjust CV_ROOT if your data lives elsewhere.
    local cv_root="${CV_ROOT:-/var/tmp/soobenve_audio/mdc_data}"
    local train_tsv=$(find "$cv_root/$lang" -name "train.tsv" 2>/dev/null | head -1)
    if [ -z "$train_tsv" ]; then
      train_tsv=$(find "$cv_root/$lang" -name "validated.tsv" 2>/dev/null | head -1)
    fi
    if [ -z "$train_tsv" ]; then
      echo "  ERROR: no train.tsv or validated.tsv found for $lang under $cv_root"
      continue
    fi

    echo "  Sweeping thresholds for $lang (tokens=$tokens_file)..."
    python "$B45_SWEEP" \
      --tokens "$tokens_file" \
      --train_tsv "$train_tsv" \
      --language "$lang" \
      --output_dir "$dir/$lang" \
      --thresholds 0.0 0.3 0.5 0.7 1.0 \
      > "$dir/${lang}_sweep.log" 2>&1
  done
}

# =============================================================================
# Tier 2a: Data-subsampling learning curves
# =============================================================================
tier2a_data_curve() {
  echo "=== Tier 2a: data-subsampling curves (10 runs) ==="
  local dir="$OUT/data_curve"
  mkdir -p "$dir"

  # Fractions chosen to produce rough approximations of CV training volumes:
  #  0.02 ≈ 15 min (at ~12h full ncx)
  #  0.05 ≈ 30 min
  #  0.10 ≈ 1 hour
  #  0.25 ≈ 2 hours
  #  0.50 ≈ 4 hours
  #  1.00 = full
  # (Exact minute counts depend on actual CV duration per language — the
  #  fractions are deterministic, the minute labels are approximate.)
  {
    for frac in 0.02 0.05 0.10 0.25 0.50 1.00; do
      for lang in sei ncx; do
        local frac_tag=$(echo "$frac" | tr . p)   # "0.10" → "0p10"
        local run_dir="$dir/${lang}_frac${frac_tag}"
        [ -d "$run_dir/adapter" ] && { echo "# skip existing: $run_dir"; continue; }
        echo "python $TRAIN --language $lang --data_fraction $frac \
              --output_dir $run_dir \
              --wandb_run_name ${lang}-frac${frac_tag} \
              > $run_dir.log 2>&1"
      done
    done
  } | run_pool
}

# =============================================================================
# Tier 2b: Model-size sweep
# =============================================================================
tier2b_size_sweep() {
  echo "=== Tier 2b: model-size sweep (6 runs) ==="
  local dir="$OUT/size_sweep"
  mkdir -p "$dir"

  # whisper-medium needs smaller per-device batch.
  {
    for size in tiny base small medium; do
      for lang in sei ncx; do
        local run_dir="$dir/${lang}_${size}"
        [ -d "$run_dir/adapter" ] && { echo "# skip existing: $run_dir"; continue; }
        local bsz=16; local gacc=1
        if [ "$size" = "medium" ]; then bsz=4; gacc=4; fi  # keep effective batch 16
        echo "python $TRAIN --language $lang --base_model $size \
              --per_device_batch $bsz --grad_accum $gacc \
              --output_dir $run_dir \
              --wandb_run_name ${lang}-${size} \
              > $run_dir.log 2>&1"
      done
    done
  } | run_pool
}

# =============================================================================
# Tier 2c: Joint with ncx upsampling
# =============================================================================
tier2c_joint_upsampled() {
  echo "=== Tier 2c: joint adapter with ncx upsampling (1 run) ==="
  echo ""
  echo "IMPORTANT: train_whisper_lora.py does NOT currently implement joint"
  echo "training with balanced sampling. This experiment requires:"
  echo "  1. A new data-loading path that concatenates sei + ncx"
  echo "  2. An upsample_minority flag that repeats the ncx dataset until"
  echo "     its weight matches sei's"
  echo "  3. Language-stratified evaluation at the end"
  echo ""
  echo "Implementing these is a one-day addition to train_whisper_lora.py."
  echo "Skipping for now — document as 'joint upsampling deferred to"
  echo "follow-up work' in the paper if not implemented."
  echo ""
  echo "If you do implement it, the expected command is:"
  echo "  python $TRAIN --language both --upsample_minority ncx \\"
  echo "    --output_dir $OUT/joint_upsampled/joint_upsample \\"
  echo "    --wandb_run_name joint-upsample"
}

# =============================================================================
# Eval all adapters with eval_adapter.py (produces test_predictions.csv in the
# schema eval_morpheme_f1.py expects, plus test_tokens.jsonl for any downstream
# B4.5 sweeps)
# =============================================================================
eval_all_adapters() {
  echo "=== Post-training evaluation: running eval_adapter.py on every adapter ==="
  local pending=()

  # Walk every experiments/ subdir that has an adapter/ child but no
  # test_predictions.csv yet. Queue them.
  while IFS= read -r adapter_dir; do
    local run_dir="$(dirname "$adapter_dir")"
    if [ -f "$run_dir/test_predictions.csv" ] && [ -f "$run_dir/test_tokens.jsonl" ]; then
      continue
    fi
    # Infer language from the directory name
    local base=$(basename "$run_dir")
    local lang
    if [[ "$base" == sei* ]]; then lang=sei
    elif [[ "$base" == ncx* ]]; then lang=ncx
    else
      echo "# skipping $run_dir: cannot infer language from name"
      continue
    fi
    # Infer base_model from the run_config.json if present
    local bm="small"
    if [ -f "$run_dir/run_config.json" ]; then
      bm=$(python -c "import json; print(json.load(open('$run_dir/run_config.json')).get('base_model','small'))" 2>/dev/null || echo small)
    fi
    pending+=("python $EVAL --adapter $adapter_dir --language $lang \
              --base_model $bm --output_dir $run_dir \
              > $run_dir/eval.log 2>&1")
  done < <(find "$OUT" -type d -name adapter)

  echo "  found ${#pending[@]} adapters to evaluate"
  printf '%s\n' "${pending[@]}" | run_pool
}

# =============================================================================
# Dispatch
# =============================================================================
PHASE="${1:-all}"

case "$PHASE" in
  tier1)
    tier1a_seed_replicates
    tier1b_rank_ablation
    tier1c_placement
    eval_all_adapters
    tier1d_b45_sweep
    ;;
  tier2)
    tier2a_data_curve
    tier2b_size_sweep
    tier2c_joint_upsampled
    eval_all_adapters
    ;;
  all)
    tier1a_seed_replicates
    tier1b_rank_ablation
    tier1c_placement
    tier2a_data_curve
    tier2b_size_sweep
    tier2c_joint_upsampled
    eval_all_adapters
    tier1d_b45_sweep
    ;;
  seed_replicates)   tier1a_seed_replicates; eval_all_adapters;;
  rank_ablation)     tier1b_rank_ablation; eval_all_adapters;;
  placement)         tier1c_placement; eval_all_adapters;;
  b45_sweep)         tier1d_b45_sweep;;
  data_curve)        tier2a_data_curve; eval_all_adapters;;
  size_sweep)        tier2b_size_sweep; eval_all_adapters;;
  joint_upsampled)   tier2c_joint_upsampled;;
  eval)              eval_all_adapters;;
  *)
    echo "unknown phase: $PHASE"
    echo "valid: tier1 tier2 all | seed_replicates rank_ablation placement b45_sweep \\"
    echo "       data_curve size_sweep joint_upsampled | eval"
    exit 1
    ;;
esac

echo ""
echo "=== Done. Outputs in $OUT ==="
