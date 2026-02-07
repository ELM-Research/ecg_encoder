#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

COMMON_ARGS=(
    --data_representation "signal"
    --objective "rectified_flow"
    --neural_network "trans_continuous_dit"
    --task "classification"
    --batch_size 2
    --epochs 2
    --ref_global_bs 16
    --distributed
    --nn_ckpt src/runs/pretrain/trans_continuous_dit/3/checkpoints/epoch_best.pt
    --torch_compile
    --add_task_head
    --ema
    --wandb
)

declare -A EXPERIMENTS=(
    ["batch9"]="qrs_dur_wide v_pacing lbbb"
    ["batch11"]="qrs_dur_wide v_pacing lbbb"
    ["batch12"]="v_pacing"
    ["batch13"]="lbbb"
    ["batch14"]="qrs_dur_wide"
)

for data in "${!EXPERIMENTS[@]}"; do
    for label in ${EXPERIMENTS[$data]}; do
        echo "=== Running: data=$data label=$label ==="
        uv run torchrun --standalone --nproc_per_node=8 \
            src/finetune_encoder.py \
            --data "$data" \
            --batch_labels "$label" \
            "${COMMON_ARGS[@]}"
    done
done
