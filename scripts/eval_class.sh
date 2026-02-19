#!/bin/bash

GPU=0

COMMON_ARGS=(
    --data_representation "signal"
    --objective "ddpm"
    --neural_network "trans_continuous_dit"
    --task "generation"
    --ema
)

COMMON_ARGS=(
    --data_representation "signal"
    --objective "recitified_flow"
    --neural_network "trans_continuous_dit"
    --task "generation"
    --ema
)

COMMON_ARGS=(
    --data_representation "signal"
    --objective "recitified_flow"
    --neural_network "trans_continuous_dit"
    --task "reconstruction"
    --ema
)

COMMON_ARGS=(
    --data_representation "signal"
    --objective "ddpm"
    --neural_network "trans_continuous_dit"
    --task "reconstruction"
    --ema
)

run_eval() {
    CUDA_VISIBLE_DEVICES=$GPU uv run src/eval_encoder.py \
        --data "$1" \
        --nn_ckpt "src/runs/pretrain/trans_continuous_dit/2/checkpoints/epoch_best.pt" \
        "${COMMON_ARGS[@]}"
}

# data    ckpt  label
run_eval mimic_iv