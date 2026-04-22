#!/usr/bin/env bash
# =============================================================================
# Hyperparameter Sensitivity Analysis (Figure 4 in paper)
# Sweeps 4 hyperparameters on Toys&Games (7B multi-view), 3 non-default
# values each (12 runs total). Default: lambda=0.10, tau=0.05, beta=3.0,
# beta_infer=0.6.
# Usage: bash run_scripts/run_sensitivity_all.sh [GPU_ID]
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${1:-${GPU_ID:-0}}
SEED=${SEED:-2025}
CONFIG="configs/toys/sasrec_align_multi_view_v3_toys_stratified_7b.yaml"

run_sensitivity() {
    local name="$1"
    local align_weight="$2"
    local tau="$3"
    local cold_boost="$4"
    local infer_boost="$5"
    local align_grid="$6"
    local tau_grid="$7"

    echo "================================================================"
    echo " Sensitivity: ${name}"
    echo " align=${align_weight}, tau=${tau}, cold=${cold_boost}, infer=${infer_boost}"
    echo "================================================================"

    python scripts/two_phase_train.py \
        --model SASRecAlignMultiViewV3 \
        --dataset Amazon_Toys_and_Games \
        --config_files "${CONFIG}" \
        --config_dict "{'align_weight': ${align_weight}, 'cold_text_boost': ${cold_boost}, 'infer_boost': ${infer_boost}, 'cold_threshold': 10}" \
        --gpu_id "${GPU_ID}" \
        --phase_a_grid \
        --align_grid "${align_grid}" \
        --tau_grid "${tau_grid}" \
        --backbone_burnin_epochs 0 \
        --burnin_eval_step 2 \
        --phase_a_epochs 20 \
        --phase_a_eval_step 1 \
        --phase_a_valid_metric "MRR@10" \
        --metric_baseline 0.0272 \
        --metric_gain_threshold 0.01 \
        --lr_text_head 2e-3 \
        --lr_dnn_cross 5e-4 \
        --phase_a_auto_to_b \
        --phase_b_epochs 40 \
        --backbone_lr_scale 0.1 \
        --checkpoint_dir "./saved/sensitivity_${name}" \
        --seed "${SEED}" \
        --variant_features "sensitivity,${name},toys,stratified" \
        --watchdog_disable \
        --save
}

echo "=============================================="
echo " MV-Align: Sensitivity Analysis (Toys, 7B)"
echo " GPU: ${GPU_ID} | Seed: ${SEED}"
echo "=============================================="

# --- Alignment weight (lambda) sweep ---
#   Default: 0.10; sweep: 0.05, 0.15, 0.20
run_sensitivity "align_005" 0.05 0.05 3.0 0.6 "0.05" "0.05"
run_sensitivity "align_015" 0.15 0.05 3.0 0.6 "0.15" "0.05"
run_sensitivity "align_020" 0.20 0.05 3.0 0.6 "0.20" "0.05"

# --- Temperature (tau) sweep ---
#   Default: 0.05; sweep: 0.03, 0.07, 0.10
run_sensitivity "tau_003" 0.1 0.03 3.0 0.6 "0.10" "0.03"
run_sensitivity "tau_007" 0.1 0.07 3.0 0.6 "0.10" "0.07"
run_sensitivity "tau_010" 0.1 0.10 3.0 0.6 "0.10" "0.10"

# --- Cold-start training boost (beta) sweep ---
#   Default: 3.0; sweep: 2.0, 4.0, 5.0
run_sensitivity "cold_20" 0.1 0.05 2.0 0.6 "0.10" "0.05"
run_sensitivity "cold_40" 0.1 0.05 4.0 0.6 "0.10" "0.05"
run_sensitivity "cold_50" 0.1 0.05 5.0 0.6 "0.10" "0.05"

# --- Inference boost (beta_infer) sweep ---
#   Default: 0.6; sweep: 0.3, 0.9, 1.2
run_sensitivity "infer_03" 0.1 0.05 3.0 0.3 "0.10" "0.05"
run_sensitivity "infer_09" 0.1 0.05 3.0 0.9 "0.10" "0.05"
run_sensitivity "infer_12" 0.1 0.05 3.0 1.2 "0.10" "0.05"

echo ""
echo "=============================================="
echo "All sensitivity experiments completed."
echo "=============================================="
