# AGENTS.md

Instructions for AI coding agents working on this repository.

## Rules

- Max 150 characters per line. Match existing style.
- Every line must earn its keep — no dead code, no narrating comments.
- Never mix functionality changes with whitespace changes.
- All functionality changes must be tested.
- Prefer editing existing files over creating new ones.

## Architecture

Two entry points, both run from the repo root:

| Script | Purpose |
|--------|---------|
| `src/pretrain_encoder.py` | Pretrain an ECG neural network |
| `src/eval_encoder.py` | Evaluate on downstream tasks (generation, reconstruction, forecasting) |

Core modules under `src/`:

| Module | Responsibility |
|--------|---------------|
| `configs/` | Argument parsing (`config.py`) and constants (`constants.py`) |
| `dataloaders/` | Dataset loading, data representations (`signal`, `bpe_symbolic`), task transforms |
| `neural_networks/` | Model architectures — add new models here with a `build_nn.py` entry |
| `optimizers/` | Optimizer factory, LR scheduling, EMA |
| `runners/` | Training loop (`train.py`) and evaluation tasks (`tasks/`) |
| `utils/` | Checkpointing, metrics, distributed setup, visualization |

## Patterns to Follow

- **Adding a model:** Create a module under `neural_networks/`, register it in `build_nn.py` and `constants.py`.
- **Adding a task:** Create a handler under `dataloaders/task/` and an eval runner under `runners/tasks/`.
- **Adding a representation:** Create a module under `dataloaders/data_representation/`, wire it in `dataset_mixer.py`.

## Stack

- Python 3.11+, `uv` for dependency management
- PyTorch with DDP for distributed training
- Rust (via `maturin`) for the BPE tokenizer only
- `wandb` for experiment tracking

## Testing

```bash
bash scripts/run_tests.sh
```
