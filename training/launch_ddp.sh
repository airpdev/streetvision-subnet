#!/usr/bin/env bash
# Launch ConvNeXt-V2-Large fine-tune across 2x A100 40GB.
# Usage:  bash training/launch_ddp.sh [extra_args...]
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/convnextv2-roadwork}"
BACKBONE="${BACKBONE:-facebook/convnextv2-large-22k-384}"
IMAGE_SIZE="${IMAGE_SIZE:-384}"
BATCH_SIZE="${BATCH_SIZE:-32}"     # per-device. 2x A100 -> effective 64
GRAD_ACCUM="${GRAD_ACCUM:-1}"
EPOCHS="${EPOCHS:-12}"
LR_BACKBONE="${LR_BACKBONE:-1e-5}"
LR_HEAD="${LR_HEAD:-1e-3}"
WD="${WD:-0.05}"
WARMUP="${WARMUP:-0.06}"
LS="${LS:-0.05}"
NUM_WORKERS="${NUM_WORKERS:-8}"

echo "[launch] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[launch] BACKBONE=${BACKBONE}"
echo "[launch] IMAGE_SIZE=${IMAGE_SIZE} BATCH_SIZE=${BATCH_SIZE} EPOCHS=${EPOCHS}"

accelerate launch --num_processes 2 --mixed_precision bf16 \
    training/train.py \
    --output_dir "${OUTPUT_DIR}" \
    --backbone "${BACKBONE}" \
    --image_size "${IMAGE_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --epochs "${EPOCHS}" \
    --lr_backbone "${LR_BACKBONE}" \
    --lr_head "${LR_HEAD}" \
    --weight_decay "${WD}" \
    --warmup_ratio "${WARMUP}" \
    --label_smoothing "${LS}" \
    --num_workers "${NUM_WORKERS}" \
    --bf16 \
    --use_class_balanced_sampler \
    "$@"
