#!/usr/bin/env bash
# =============================================================================
# Main Table Experiments (Table 2 in paper)
# Reproduces all 8 configurations x 4 seeds for Beauty and Toys datasets.
# Usage: bash run_scripts/run_main_table.sh [GPU_ID]
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GPU_ID=${1:-${GPU_ID:-0}}
SEEDS=(42 2024 2025 2026)

run_experiment() {
    local model="$1"
    local dataset="$2"
    local config="$3"
    local ckpt_dir="$4"
    local variant="$5"
    local seed="$6"

    echo "================================================================"
    echo " Model: ${model} | Dataset: ${dataset} | Seed: ${seed}"
    echo " Config: ${config}"
    echo "================================================================"

    python scripts/two_phase_train.py \
        --model "${model}" \
        --dataset "${dataset}" \
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
        --checkpoint_dir "./saved/${ckpt_dir}" \
        --seed "${seed}" \
        --variant_features "${variant}" \
        --watchdog_disable \
        --save
}

EXPERIMENTS=(
    # model | dataset | config_file | checkpoint_subdir | variant_features
    "SASRecAlignV3|Amazon_Beauty|configs/beauty/sasrec_align_base_stratified_v3.yaml|tfidf_v3_beauty|sasrec,tfidf,v3,beauty,stratified"
    "SASRecAlignV3|Amazon_Beauty|configs/beauty/sasrec_align_qwen3_stratified_v3.yaml|tfidf_llm_v3_beauty|sasrec,tfidf,llm,v3,beauty,stratified"
    "SASRecAlignMultiViewV3|Amazon_Beauty|configs/beauty/sasrec_align_multi_view_v3_stratified.yaml|multiview_v3_7b_beauty|sasrec,multiview_v3,7b,4views,beauty,stratified"
    "SASRecAlignMultiViewV3|Amazon_Beauty|configs/beauty/sasrec_align_multi_view_v3_stratified_14b.yaml|multiview_v3_14b_beauty|sasrec,multiview_v3,14b,4views,beauty,stratified"
    "SASRecAlignV3|Amazon_Toys_and_Games|configs/toys/sasrec_align_toys_base_stratified_v3.yaml|tfidf_v3_toys|sasrec,tfidf,v3,toys,stratified"
    "SASRecAlignV3|Amazon_Toys_and_Games|configs/toys/sasrec_align_toys_qwen3_stratified_v3.yaml|tfidf_llm_v3_toys|sasrec,tfidf,llm,v3,toys,stratified"
    "SASRecAlignMultiViewV3|Amazon_Toys_and_Games|configs/toys/sasrec_align_multi_view_v3_toys_stratified_7b.yaml|multiview_v3_7b_toys|sasrec,multiview_v3,7b,toys,stratified"
    "SASRecAlignMultiViewV3|Amazon_Toys_and_Games|configs/toys/sasrec_align_multi_view_v3_toys_stratified_14b.yaml|multiview_v3_14b_toys|sasrec,multiview_v3,14b,toys,stratified"
)

echo "=============================================="
echo " MV-Align: Main Table Experiments"
echo " GPU: ${GPU_ID} | Seeds: ${SEEDS[*]}"
echo "=============================================="

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r model dataset config ckpt variant <<< "${exp}"
    for seed in "${SEEDS[@]}"; do
        run_experiment "${model}" "${dataset}" "${config}" "${ckpt}_seed${seed}" "${variant}" "${seed}"
    done
done

echo ""
echo "=============================================="
echo "All main table experiments completed."
echo "=============================================="
