CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=8 \
src/pretrain_encoder.py \
--data batch15 batch17 batch18 \
--data_representation "signal" \
--objective "ddpm" \
--neural_network "trans_continuous_dit" \
--task "pretrain" \
--batch_size 32 \
--distributed \
--ref_global_bs 256 \
--epochs 50 \
--torch_compile \
--ema \
--lr 1e-4 \
--lr_schedule constant \
--beta1 0.9 \
--beta2 0.999 \
--weight_decay 1e-4 \
--augment \
--warmup 200 \
--optimizer adamw \
--grad_clip 1.0 \
--lr_schedule cosine \
--wandb

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=8 \
src/pretrain_encoder.py \
--data batch15 batch17 batch18 \
--data_representation "signal" \
--objective "rectified_flow" \
--neural_network "trans_continuous_dit" \
--task "pretrain" \
--batch_size 32 \
--distributed \
--ref_global_bs 256 \
--epochs 50 \
--torch_compile \
--ema \
--lr 1e-4 \
--lr_schedule constant \
--beta1 0.9 \
--beta2 0.999 \
--weight_decay 1e-4 \
--augment \
--warmup 200 \
--optimizer adamw \
--grad_clip 1.0 \
--wandb


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=8 \
src/pretrain_encoder.py \
--data mimic_iv ptb_xl code15 csn cpsc \
--data_representation "signal" \
--objective "ddpm" \
--neural_network "trans_continuous_dit" \
--task "pretrain" \
--batch_size 32 \
--distributed \
--ref_global_bs 256 \
--epochs 50 \
--torch_compile \
--ema \
--lr 1e-4 \
--lr_schedule constant \
--beta1 0.9 \
--beta2 0.999 \
--weight_decay 1e-4 \
--augment \
--warmup 800 \
--optimizer adamw \
--grad_clip 1.0 \
--wandb

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=8 \
src/pretrain_encoder.py \
--data mimic_iv ptb_xl code15 csn cpsc \
--data_representation "signal" \
--objective "rectified_flow" \
--neural_network "trans_continuous_dit" \
--task "pretrain" \
--batch_size 32 \
--distributed \
--ref_global_bs 256 \
--epochs 50 \
--torch_compile \
--ema \
--lr 1e-4 \
--lr_schedule constant \
--beta1 0.9 \
--beta2 0.999 \
--weight_decay 1e-4 \
--augment \
--warmup 800 \
--optimizer adamw \
--grad_clip 1.0 \
--wandb

# Muon optimizer examples (requires PyTorch >= 2.9)
# Muon uses orthogonalization for 2D hidden weights; AdamW for rest
# Reference: https://arxiv.org/abs/2510.19376 (18% lower loss than AdamW on diffusion)

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=8 \
src/pretrain_encoder.py \
--data batch15 batch17 batch18 \
--data_representation "signal" \
--objective "ddpm" \
--neural_network "trans_continuous_dit" \
--task "pretrain" \
--batch_size 32 \
--distributed \
--ref_global_bs 256 \
--epochs 50 \
--torch_compile \
--ema \
--lr 2e-2 \
--lr_schedule cosine \
--weight_decay 1e-2 \
--augment \
--warmup 500 \
--optimizer muon \
--muon_momentum 0.95 \
--muon_adamw_lr_ratio 0.1 \
--grad_clip 1.0 \
--wandb

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=8 \
src/pretrain_encoder.py \
--data batch15 batch17 batch18 \
--data_representation "signal" \
--objective "rectified_flow" \
--neural_network "trans_continuous_dit" \
--task "pretrain" \
--batch_size 32 \
--distributed \
--ref_global_bs 256 \
--epochs 50 \
--torch_compile \
--ema \
--lr 2e-2 \
--lr_schedule cosine \
--weight_decay 1e-2 \
--augment \
--warmup 500 \
--optimizer muon \
--muon_momentum 0.95 \
--muon_adamw_lr_ratio 0.1 \
--grad_clip 1.0 \
--wandb
