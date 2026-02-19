# CLAUDE.md

## Project

ecg_encoder is a research framework for pretraining and evaluating ECG neural networks. The audience is ML researchers iterating on ECG representation learning.

## Philosophy

Every line must earn its keep. Prefer readability over cleverness. If carefully designed, 10 lines can have the impact of 1000. Never mix functionality changes with whitespace changes. All functionality changes must be tested.

## Architecture

```
src/
├── pretrain_encoder.py          # Entry: pretraining
├── eval_encoder.py              # Entry: downstream evaluation
├── configs/                     # Arg parsing, constants
├── dataloaders/                 # Dataset, representations (signal, bpe_symbolic), tasks
├── neural_networks/             # Model architectures (DiT, NEPA, decoder, MAE, MERL, etc.)
├── optimizers/                  # LR scheduling, EMA
├── runners/                     # Training loop, eval tasks (generation, reconstruction, forecasting)
└── utils/                       # Checkpointing, metrics, distributed, viz
```

## Conventions

- Python 3.11+, managed with `uv`
- Max 150 characters per line, match existing style
- No comments that merely narrate what code does
- Entry points run from repo root: `uv run torchrun ... src/pretrain_encoder.py`

## Key Patterns

- **Factory:** `BuildNN`, `BuildDataLoader` construct components from args
- **Strategy:** task handlers (`Pretrain`, `Forecasting`), data representations (`Signal`, `BPESymbolic`)
- **Config dataclasses:** model hyperparameters (e.g., `DiTConfig`, `DecoderTransformerConfig`)
- **DDP:** distributed training via `utils/gpu_setup.py`

## Testing

```bash
bash scripts/run_tests.sh
```

## Dependencies

Managed in `pyproject.toml`. Install with `uv sync`. BPE tokenizer requires Rust — compile with `maturin develop --release` in `src/dataloaders/data_representation/bpe/`.
