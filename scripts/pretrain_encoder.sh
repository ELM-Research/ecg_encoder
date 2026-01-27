CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 \
uv run torchrun --standalone --nproc_per_node=8 \
src/pretrain_encoder.py \
--data mimic_iv ptb_xl code15 csn cpsc \
--data_representation "bpe_symbolic" \
--objective "autoregressive" \
--neural_network "trans_discrete_decoder" \
--task "pretrain" \
--batch_size 12 \
--distributed \
--ref_global_bs 96 \
--augment \
--bpe_symbolic_len 4096 \
--epochs 10 \
--torch_compile \
--warmup 2500 \
--bfloat_16 \
--wandb

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 \
uv run torchrun --standalone --nproc_per_node=8 \
src/pretrain_encoder.py \
--data mimic_iv ptb_xl code15 csn cpsc \
--data_representation "bpe_symbolic" \
--objective "autoregressive" \
--neural_network "trans_discrete_decoder" \
--task "pretrain" \
--batch_size 12 \
--distributed \
--ref_global_bs 96 \
--augment \
--bpe_symbolic_len 4096 \
--epochs 10 \
--torch_compile \
--warmup 2500 \
--wandb


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 \
# uv run torchrun --standalone --nproc_per_node=8 \
# src/pretrain_encoder.py \
# --data mimic_iv ptb_xl code15 csn cpsc \
# --data_representation "signal" \
# --objective "ddpm" \
# --neural_network "trans_continuous_dit" \
# --task "pretrain" \
# --batch_size 64 \
# --distributed \
# --ref_global_bs 512 \
# --augment \
# --epochs 10 \
# --warmup 2500 \
# --torch_compile \
# --wandb

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 \
# uv run torchrun --standalone --nproc_per_node=8 \
# src/pretrain_encoder.py \
# --data mimic_iv ptb_xl code15 csn cpsc \
# --data_representation "signal" \
# --objective "rectified_flow" \
# --neural_network "trans_continuous_dit" \
# --task "pretrain" \
# --batch_size 64 \
# --distributed \
# --ref_global_bs 512 \
# --augment \
# --epochs 10 \
# --warmup 2500 \
# --torch_compile \
# --wandb

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 \
# uv run torchrun --standalone --nproc_per_node=8 \
# src/pretrain_encoder.py \
# --data mimic_iv ptb_xl code15 csn cpsc \
# --data_representation "signal" \
# --objective "rectified_flow" \
# --neural_network "trans_continuous_dit" \
# --task "pretrain" \
# --batch_size 64 \
# --distributed \
# --ref_global_bs 512 \
# --augment \
# --epochs 2 \
# --warmup 500 \
# --wandb



# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 \
# uv run torchrun --standalone --nproc_per_node=8 \
# src/pretrain_encoder.py \
# --data mimic_iv ptb_xl code15 csn cpsc \
# --data_representation "signal" \
# --objective "mae" \
# --neural_network "mae_vit" \
# --task "pretrain" \
# --batch_size 64 \
# --distributed \
# --ref_global_bs 512 \
# --augment \
# --epochs 2 \
# --warmup 500 \
# --wandb