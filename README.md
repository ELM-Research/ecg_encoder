# ecg_encoder

## Installation

1. `git clone https://github.com/ELM-Research/ecg_encoder.git`

2. `cd` into repo and `uv sync`.

## Overview

This is a research-oriented, open-source repository for pretraining and evaluation of ECG-specific neural networks. Currently, this repository is in beta.

## Data

Currently, we support the following dataset (see `BASE_DATASETS` from `src/configs/constants.py`):

1. PTB-XL
2. MIMIC-IV-ECG
3. CODE 15
4. CPSC
5. CSN

Please first use [https://github.com/ELM-Research/ecg_preprocess](https://github.com/ELM-Research/ecg_preprocess) to prepare these datasets and set `DATA_DIR` from `src/configs/constants.py` accordingly.

## ECG Input Representations
This current repository considers 2 input representations of ECGs as defined below:

**ECG Signal (`--data_representation signal`):**  
The raw ECG signal is represented as a matrix `X_sig` $\in$ `R^(C x L)`, where `C` denotes the number of leads and `L` is the number of time samples per lead. All other modalities are derived from `X_sig`. This is the most common way of representing ECGs.

**ECG Symbol (`--data_representation bpe_symbolic`):**  
We use ECG-Byte’s compression schema to convert ECG signals into text. First, a normalized and discretized ECG signal `X_sig` is mapped to a symbolic sequence using a set of symbols `A = {a, b, …, z}`. This sequence is then flattened into a one-dimensional array `X_symb` $\in$ `A^(C * L)`. Finally, a byte-pair encoding (BPE) process compresses `X_symb` into a sequence of tokens from an extended vocabulary `V`, resulting in the final textual representation `X_ID` $\in$ `V^(m)`, where `m` is the length of the token sequence. You can train your own tokenizer on [https://github.com/ELM-Research/ecg_preprocess](https://github.com/ELM-Research/ecg_preprocess) or use our provided tokenizer as is in `src/dataloaders/data_representation/bpe/ecg_byte_tokenizer_10000.pkl`. Please make sure to compile the bpe tokenizer by `cd src/dataloaders/data_representation/bpe` and running `maturin develop --release`. If rust is not installed do `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain=1.82.0 -y`.

## Pretraining

1. Pretraining DiT with DDPM

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=8 \
    src/pretrain_encoder.py \
    --data mimic_iv \
    --data_representation "signal" \
    --objective ddpm \
    --neural_network "trans_continuous_dit" \
    --task "pretrain" \
    --batch_size 64 \
    --distributed \
    --ema
```

2. Pretraining DiT with Rectified Flow

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=8 \
    src/pretrain_encoder.py \
    --data mimic_iv \
    --data_representation "signal" \
    --objective rectified_flow \
    --neural_network "trans_continuous_dit" \
    --task "pretrain" \
    --batch_size 64 \
    --distributed \
    --ema
```

3. Pretraining NEPA

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=8 \
    src/pretrain_encoder.py \
    --data mimic_iv \
    --data_representation "signal" \
    --objective autoregressive \
    --neural_network "trans_continuous_nepa" \
    --task "pretrain" \
    --batch_size 64 \
    --distributed \
    --ema
```

4. Pretraining Decoder-only Transformer

```
CUDA_VISIBLE_DEVICES=4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=4 \
src/pretrain_encoder.py \
--data mimic_iv \
--data_representation "bpe_symbolic" \
--objective "autoregressive" \
--neural_network "trans_discrete_decoder" \
--task "pretrain" \
--batch_size 64 \
--distributed
```

5. Pretraining MAE VIT

```
CUDA_VISIBLE_DEVICES=4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=4 \
src/pretrain_encoder.py \
--data mimic_iv \
--data_representation "signal" \
--objective "mae" \
--neural_network "mae_vit" \
--task "pretrain" \
--batch_size 64 \
--distributed
```

6. Pretraining MERL

```
CUDA_VISIBLE_DEVICES=4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=4 \
src/pretrain_encoder.py \
--data mimic_iv \
--data_representation "signal" \
--objective "merl" \
--neural_network "merl" \
--task "pretrain" \
--batch_size 64 \
--text_feature_extractor ncbi/MedCPT-Query-Encoder \
--distributed
```

## Downstream Tasks

1. Forecasting Decoder-only Transformer

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=8 \
    src/eval_encoder.py
    --data mimic_iv \
    --nn_ckpt $PATH_TO_PRETRAINED_CKPT \
    --data_representation "bpe_symbolic"
    --objective "autoregressive"
    --neural_network "trans_discrete_decoder"
    --task "forecasting"
    --forecast_ratio 0.5
    --bpe_symbolic_len 2048
    --num_workers 4
```

## License <a name="license"></a>

This repository contains code licensed under the MIT License, except for the following `.py` files in the `ecg_bench/models/encoder` directory: `st_mem.py`, `mlae.py`, `mtae.py`. These files are licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. Please view the original license in their [respective repository](https://github.com/bakqui/ST-MEM?tab=License-1-ov-file#readme) for more details.