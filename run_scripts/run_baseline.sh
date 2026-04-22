#!/usr/bin/env bash
# =============================================================================
# ID-only Baseline (SASRec, 50 epochs)
# Reproduces the SASRec (ID-only) rows in Table 2.
# Usage: bash run_scripts/run_baseline.sh [GPU_ID]
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

GPU_ID=${1:-${GPU_ID:-0}}
SEEDS=(42 2024 2025 2026)

echo "=============================================="
echo " SASRec ID-only Baseline (50 epochs)"
echo " GPU: ${GPU_ID} | Seeds: ${SEEDS[*]}"
echo "=============================================="

BASELINES=(
    "Amazon_Beauty|configs/beauty/sasrec_baseline_50ep_stratified.yaml|baseline_beauty"
    "Amazon_Toys_and_Games|configs/toys/sasrec_baseline_50ep_toys_stratified.yaml|baseline_toys"
)

for bl in "${BASELINES[@]}"; do
    IFS='|' read -r dataset config ckpt <<< "${bl}"
    for seed in "${SEEDS[@]}"; do
        echo ""
        echo "--- ${dataset} | seed=${seed} ---"
        python scripts/two_phase_train.py \
            --model SASRecAlign \
            --dataset "${dataset}" \
            --config_files "${config}" \
            --gpu_id "${GPU_ID}" \
            --only_phase_a \
            --phase_a_epochs 50 \
            --phase_a_eval_step 5 \
            --phase_a_valid_metric "MRR@10" \
            --checkpoint_dir "./saved/${ckpt}_seed${seed}" \
            --seed "${seed}" \
            --variant_features "sasrec,idonly,${dataset},stratified" \
            --watchdog_disable \
            --save
    done
done

echo ""
echo "=============================================="
echo "All baseline experiments completed."
echo "=============================================="
