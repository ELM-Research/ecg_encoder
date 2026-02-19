#!/bin/bash

GPU=1

COMMON_ARGS=(
    --data_representation "bpe_symbolic"
    --objective "autoregressive"
    --neural_network "trans_discrete_decoder"
    --task "forecasting"
    --forecast_ratio 0.5
    --bpe_symbolic_len 2048
    --num_workers 4
)

run_eval() {
    CUDA_VISIBLE_DEVICES=$GPU uv run src/eval_encoder.py \
        --data "$1" \
        --nn_ckpt "src/runs/pretrain/trans_discrete_decoder/0/checkpoints/epoch_best.pt" \
        "${COMMON_ARGS[@]}"
}

run_eval batch10

echo "All evaluations complete"