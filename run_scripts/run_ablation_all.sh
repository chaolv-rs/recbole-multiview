#!/usr/bin/env bash
# =============================================================================
# Ablation Study (Table 3 in paper)
# All ablation experiments on Amazon Toys&Games with Qwen2.5-7B.
# Covers: -SE, -Cross, -SE-Cross, -Whiten (multi-view), -Whiten (single-view),
#         -Align (no contrastive alignment).
# Usage: bash run_scripts/run_ablation_all.sh [GPU_ID]
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${1:-${GPU_ID:-0}}
SEED=${SEED:-2025}

run_ablation() {
    local name="$1"
    local model="$2"
    local config="$3"

    echo "================================================================"
    echo " Ablation: ${name}"
    echo " Model: ${model} | Config: ${config}"
    echo "================================================================"

    python scripts/two_phase_train.py \
        --model "${model}" \
        --dataset Amazon_Toys_and_Games \
        --config_files "${config}" \
        --config_dict "{'align_weight': 0.1, 'cold_text_boost': 3.0, 'infer_boost': 0.6, 'cold_threshold': 10}" \
        --gpu_id "${GPU_ID}" \
        --phase_a_grid \
        --align_grid "0.10" \
        --tau_grid "0.05" \
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
        --checkpoint_dir "./saved/ablation_${name}" \
        --seed "${SEED}" \
        --variant_features "ablation,${name},toys,stratified" \
        --watchdog_disable \
        --save
}

echo "=============================================="
echo " MV-Align: Ablation Study (Toys, 7B)"
echo " GPU: ${GPU_ID} | Seed: ${SEED}"
echo "=============================================="

# RQ2/RQ4: Remove SE-style gating + Cross network
run_ablation "nosenet_nocross" \
    "SASRecAlignMultiViewV3" \
    "configs/ablation/sasrec_align_multi_view_v3_toys_stratified_7b_nocross_nosenet.yaml"

# RQ2: Remove SE-style gating only
run_ablation "nosenet" \
    "SASRecAlignMultiViewV3" \
    "configs/ablation/sasrec_align_multi_view_v3_toys_stratified_7b_nosenet.yaml"

# RQ4: Remove Cross network only
run_ablation "nocross" \
    "SASRecAlignMultiViewV3" \
    "configs/ablation/sasrec_align_multi_view_v3_toys_stratified_7b_nocross.yaml"

# RQ1: Multi-view without whitening
run_ablation "multiview_no_whiten" \
    "SASRecAlignMultiViewV3" \
    "configs/ablation/sasrec_align_multi_view_v3_toys_stratified_7b_no_whiten.yaml"

# RQ1: Single-view (TF-IDF+LLM) without whitening
run_ablation "singleview_no_whiten" \
    "SASRecAlignV3" \
    "configs/ablation/sasrec_align_toys_qwen3_stratified_v3_no_whiten.yaml"

# RQ4: Remove contrastive alignment
run_ablation "no_align" \
    "SASRecAlignMultiViewV3" \
    "configs/ablation/sasrec_align_multi_view_v3_toys_stratified_7b_no_align.yaml"

echo ""
echo "=============================================="
echo "All ablation experiments completed."
echo "=============================================="
