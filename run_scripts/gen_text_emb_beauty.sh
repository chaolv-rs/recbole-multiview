#!/usr/bin/env bash
# =============================================================================
# Generate text embeddings for Amazon Beauty
# Step 1: TF-IDF (base) embeddings
# Step 2: Export internal item mapping
# Step 3: LLM single-view embeddings (Qwen2.5-7B)
# Step 4: LLM multi-view embeddings (4 views: identity/function/audience/category)
#
# Prerequisites:
#   - Qwen2.5-7B-Instruct model weights (set LLM_MODEL_PATH)
#   - Amazon_Beauty dataset in dataset/Amazon_Beauty/
#
# Usage: LLM_MODEL_PATH=/path/to/qwen2.5-7b bash run_scripts/gen_text_emb_beauty.sh
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

LLM_MODEL_PATH="${LLM_MODEL_PATH:?Set LLM_MODEL_PATH to the Qwen2.5-7B-Instruct model directory}"
DEVICE="${DEVICE:-cuda:0}"

export OMP_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)

DATASET="Amazon_Beauty"
DATA_DIR="dataset/${DATASET}"

echo "========================================"
echo " Text Embedding Generation: ${DATASET}"
echo " LLM: ${LLM_MODEL_PATH}"
echo "========================================"

# Step 1: TF-IDF embeddings
echo "[1/4] TF-IDF (base) embeddings with center+whiten..."
python tools/build_item_text_emb_base.py \
    --dataset "${DATASET}" \
    --config configs/sasrec_base_plain.yaml \
    --output "${DATA_DIR}/item_text_emb.base.npy" \
    --title_field title \
    --svd_dim 256 \
    --svd_random_state 42 \
    --ngram_min 1 \
    --ngram_max 2 \
    --min_df 2 \
    --dtype float16
echo "  -> ${DATA_DIR}/item_text_emb.base.npy"

# Step 2: Export item mapping
echo "[2/4] Exporting item index mapping..."
python tools/export_internal_item_mapping.py \
    --dataset "${DATASET}" \
    --config configs/sasrec_base_plain.yaml \
    --output "${DATA_DIR}/item_index_mapping.csv"
echo "  -> ${DATA_DIR}/item_index_mapping.csv"

# Step 3: Single-view LLM embeddings
echo "[3/4] LLM single-view embeddings (mean pooling, 256-d)..."
python tools/build_item_text_emb_qwen3_hf.py \
    --mapping "${DATA_DIR}/item_index_mapping.csv" \
    --model_name_or_path "${LLM_MODEL_PATH}" \
    --output "${DATA_DIR}/item_text_emb.qwen2.5_7b.base.npy" \
    --prompt_preset base \
    --output_mode mean \
    --project_dim 256 \
    --dataset "${DATASET}" \
    --config configs/sasrec_base_plain.yaml recbole/properties/overall.yaml \
    --batch_size 16 \
    --max_length 0 \
    --dtype float16 \
    --device "${DEVICE}" \
    --svd_random_state 42 \
    --use_chat_template
echo "  -> ${DATA_DIR}/item_text_emb.qwen2.5_7b.base.npy"

# Step 4: Multi-view LLM embeddings (4 views x 64-d = 256-d total)
echo "[4/4] LLM multi-view embeddings (4 views, per-view 64-d)..."
python tools/build_item_text_emb_qwen3_hf.py \
    --mapping "${DATA_DIR}/item_index_mapping.csv" \
    --model_name_or_path "${LLM_MODEL_PATH}" \
    --output "${DATA_DIR}/item_text_emb.qwen2.5_7b.multiview.npy" \
    --prompt_preset multiview \
    --output_mode concat \
    --split_output_dir "${DATA_DIR}/qwen2.5_7b_4views" \
    --view_project_dim 64 \
    --dataset "${DATASET}" \
    --config configs/sasrec_base_plain.yaml recbole/properties/overall.yaml \
    --batch_size 16 \
    --max_length 0 \
    --dtype float16 \
    --device "${DEVICE}" \
    --svd_random_state 42 \
    --use_chat_template
echo "  -> ${DATA_DIR}/qwen2.5_7b_4views/view_{0..3}.npy"

echo ""
echo "========================================"
echo "All embeddings generated for ${DATASET}."
echo "========================================"
