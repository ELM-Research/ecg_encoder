#!/bin/bash


# CUDA_VISIBLE_DEVICES=0 uv run src/eval_encoder.py \
# --data batch10 \
# --data_representation "signal" \
# --objective "ddpm" \
# --neural_network "trans_continuous_dit" \
# --task "generation" \
# --nn_ckpt "src/runs/pretrain/trans_continuous_dit/1/checkpoints/epoch_best.pt" \
# --condition lead \
# --num_workers 4 \
# --ema

CUDA_VISIBLE_DEVICES=0 uv run src/eval_encoder.py \
--data batch10 \
--data_representation "signal" \
--objective "rectified_flow" \
--neural_network "trans_continuous_dit" \
--task "generation" \
--nn_ckpt "src/runs/pretrain/trans_continuous_dit/4/checkpoints/epoch_best.pt" \
--num_workers 4 \
--ema \
--text_feature_extractor Qwen/Qwen3-0.6B \
--condition text