# for objective in ddpm rectified_flow; do
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     uv run torchrun --standalone --nproc_per_node=8 \
#     src/pretrain_encoder.py \
#     --data batch9 \
#     --data_representation "signal" \
#     --objective "$objective" \
#     --neural_network "trans_continuous_dit" \
#     --task "pretrain" \
#     --batch_size 64 \
#     --distributed \
#     --ref_global_bs 512 \
#     --epochs 100 \
#     --torch_compile \
#     --ema \
#     --lr 1e-3 \
#     --lr_schedule cosine \
#     --weight_decay 3.2e-2 \
#     --beta1 0.9 \
#     --beta2 0.999 \
#     --augment \
#     --warmup 10000 \
#     --optimizer muon \
#     --grad_clip 1.0 \
#     --wandb
# done


# for objective in ddpm rectified_flow; do
#     CUDA_VISIBLE_DEVICES=4,5,6,7 \
#     uv run torchrun --standalone --nproc_per_node=4 \
#     src/pretrain_encoder.py \
#     --data mimic_iv ptb_xl batch9 \
#     --data_representation "signal" \
#     --objective "$objective" \
#     --neural_network "trans_continuous_dit" \
#     --task "pretrain" \
#     --batch_size 64 \
#     --distributed \
#     --ref_global_bs 256 \
#     --epochs 50 \
#     --torch_compile \
#     --ema \
#     --lr 1e-3 \
#     --lr_schedule cosine \
#     --weight_decay 3.2e-2 \
#     --beta1 0.9 \
#     --beta2 0.999 \
#     --augment \
#     --warmup 10000 \
#     --text_feature_extractor Qwen/Qwen3-0.6B \
#     --condition text \
#     --optimizer muon \
#     --grad_clip 1.0 \
#     --num_workers 16 \
#     --wandb
# done


for objective in rectified_flow; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    uv run torchrun --standalone --nproc_per_node=4 \
    src/pretrain_encoder.py \
    --data mimic_iv ptb_xl batch9 \
    --data_representation "signal" \
    --objective "$objective" \
    --neural_network "trans_continuous_dit" \
    --task "pretrain" \
    --batch_size 64 \
    --distributed \
    --ref_global_bs 256 \
    --epochs 15 \
    --torch_compile \
    --ema \
    --lr 1e-3 \
    --lr_schedule cosine \
    --weight_decay 3.2e-2 \
    --beta1 0.9 \
    --beta2 0.999 \
    --augment \
    --warmup 10000 \
    --condition lead \
    --optimizer muon \
    --grad_clip 1.0 \
    --num_workers 16 \
    --wandb
done

for objective in rectified_flow; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    uv run torchrun --standalone --nproc_per_node=4 \
    src/pretrain_encoder.py \
    --data mimic_iv ptb_xl batch9 \
    --data_representation "signal" \
    --objective "$objective" \
    --neural_network "trans_continuous_dit" \
    --task "pretrain" \
    --batch_size 64 \
    --distributed \
    --ref_global_bs 256 \
    --epochs 15 \
    --torch_compile \
    --ema \
    --lr 1e-3 \
    --lr_schedule cosine \
    --weight_decay 3.2e-2 \
    --beta1 0.9 \
    --beta2 0.999 \
    --augment \
    --warmup 10000 \
    --text_feature_extractor Qwen/Qwen3-0.6B \
    --condition text \
    --optimizer muon \
    --grad_clip 1.0 \
    --num_workers 16 \
    --wandb
done


CUDA_VISIBLE_DEVICES=4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=4 \
src/pretrain_encoder.py \
--data mimic_iv ptb_xl batch9 \
--data_representation "bpe_symbolic" \
--objective "autoregressive" \
--neural_network "trans_discrete_decoder" \
--task "pretrain" \
--batch_size 32 \
--distributed \
--ref_global_bs 128 \
--epochs 15 \
--torch_compile \
--lr 1e-2 \
--lr_schedule cosine \
--weight_decay 5e-2 \
--beta1 0.9 \
--beta2 0.95 \
--augment \
--warmup 40000 \
--optimizer muon \
--muon_adamw_lr_ratio 0.1 \
--bpe_symbolic_len 2048 \
--grad_clip 1.0 \
--num_workers 16 \
--wandb


CUDA_VISIBLE_DEVICES=4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=4 \
src/pretrain_encoder.py \
--data mimic_iv ptb_xl batch9 \
--data_representation "bpe_symbolic" \
--objective "autoregressive" \
--neural_network "trans_discrete_decoder" \
--task "pretrain" \
--batch_size 32 \
--distributed \
--ref_global_bs 128 \
--epochs 15 \
--torch_compile \
--lr 1e-2 \
--lr_schedule cosine \
--weight_decay 5e-2 \
--beta1 0.9 \
--beta2 0.95 \
--augment \
--warmup 40000 \
--optimizer muon \
--muon_adamw_lr_ratio 0.1 \
--bpe_symbolic_len 2048 \
--grad_clip 1.0 \
--num_workers 16 \
--bfloat_16 \
--wandb

CUDA_VISIBLE_DEVICES=4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=4 \
src/pretrain_encoder.py \
--data mimic_iv \
--data_representation "signal" \
--objective "merl" \
--neural_network "merl" \
--task "pretrain" \
--batch_size 64 \
--distributed \
--ref_global_bs 256 \
--epochs 100 \
--torch_compile \
--lr_schedule cosine \
--optimizer adamw \
--augment \
--warmup 30000 \
--text_feature_extractor ncbi/MedCPT-Query-Encoder \
--grad_clip 1.0 \
--num_workers 16 \
--wandb

CUDA_VISIBLE_DEVICES=4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=4 \
src/pretrain_encoder.py \
--data mimic_iv \
--data_representation "signal" \
--objective "mtae" \
--neural_network "mtae" \
--task "pretrain" \
--batch_size 64 \
--distributed \
--ref_global_bs 256 \
--epochs 100 \
--torch_compile \
--lr_schedule cosine \
--optimizer adamw \
--augment \
--warmup 30000 \
--grad_clip 1.0 \
--num_workers 16 \
--wandb


CUDA_VISIBLE_DEVICES=4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=4 \
src/pretrain_encoder.py \
--data mimic_iv \
--data_representation "signal" \
--objective "mlae" \
--neural_network "mlae" \
--task "pretrain" \
--batch_size 64 \
--distributed \
--ref_global_bs 256 \
--epochs 100 \
--torch_compile \
--lr_schedule cosine \
--optimizer adamw \
--augment \
--warmup 30000 \
--grad_clip 1.0 \
--num_workers 16 \
--wandb

CUDA_VISIBLE_DEVICES=4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=4 \
src/pretrain_encoder.py \
--data mimic_iv \
--data_representation "signal" \
--objective "st_mem" \
--neural_network "st_mem" \
--task "pretrain" \
--batch_size 64 \
--distributed \
--ref_global_bs 256 \
--epochs 100 \
--torch_compile \
--lr_schedule cosine \
--optimizer adamw \
--augment \
--warmup 30000 \
--grad_clip 1.0 \
--num_workers 16 \
--wandb