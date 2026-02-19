#!/bin/bash

GPU=0

COMMON_ARGS=(
    --data_representation "signal"
    --objective "ddpm"
    --neural_network "trans_continuous_dit"
    --task "generation"
    --ema
)

run_eval() {
    CUDA_VISIBLE_DEVICES=$GPU uv run src/eval_encoder.py \
        --data "$1" \
        --nn_ckpt "src/runs/pretrain/trans_continuous_dit/2/checkpoints/epoch_best.pt" \
        "${COMMON_ARGS[@]}"
}

# data    ckpt  label
run_eval batch11 5 qrs_dur_wide
run_eval batch11 6 v_pacing
run_eval batch11 7 lbbb
run_eval batch9  0 qrs_dur_wide
run_eval batch9  1 v_pacing
run_eval batch9  2 lbbb
run_eval batch14 8 qrs_dur_wide
run_eval batch12 4 v_pacing
run_eval batch13 3 lbbb

echo "All evaluations complete"