import argparse
from configs.constants import Mode, ALLOWED_DATA


def get_args(mode: Mode) -> argparse.Namespace:
    if mode not in {"pretrain", "downstream_train", "downstream_eval"}:
        raise ValueError(f"invalid mode: {mode}")

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--seed", type=int, default=0, help="Random Seed")
    parser.add_argument("--dev", action="store_true", default=None, help="Development mode")

    if mode in {"pretrain", "downstream_train", "downstream_eval"}:
        parser.add_argument("--task", type=str, default=None, choices=["pretrain", "classification", "translation", "forecasting"])
        parser.add_argument("--forecast_ratio", type=float, default=0.5, help="Please choose the percentage you want to forecast")
        parser.add_argument(
            "--data",
            type=str,
            nargs="+",
            required=True,
            help=f"One or more datasets: {', '.join(sorted(ALLOWED_DATA))}",
        )
        parser.add_argument("--data_subset", type=float, default=1, help="Please choose the percentage of the data you want to use.")
        parser.add_argument(
            "--data_sampling_probs",
            type=float,
            nargs="+",
            default=None,
            help="If one or more datasets, specify specify the probability of sampling from each dataset.",
        )
        parser.add_argument("--augment", action="store_true", default=None, help="Choose whether you want to augment your ECG")
        parser.add_argument(
            "--data_representation",
            type=str,
            default=None,
            choices=["signal", "rgb", "bpe_symbolic"],
            help="Please choose the representation of data you want to input into the neural network.",
        )
        parser.add_argument(
            "--objective",
            type=str,
            default=None,
            choices=["autoregressive", "mae", "mlm", "ddpm", "rectified_flow"],
            help="Please choose the representation of data you want to input into the neural network.",
        )
        parser.add_argument("--patch_dim", type=int, default=2500, help="Please choose a patch dim that is evenly divisible by signal_len.")
        parser.add_argument("--num_patches", type=int, default=12, help="Please choose number of patches.")
        parser.add_argument(
            "--bpe_symbolic_len", type=int, default=2048, help="Please choose the bpe symbolic len for the bpe_symbolic data representation."
        )
        parser.add_argument("--segment_len", type=int, default=2500, help="ECG Segment Length")
        parser.add_argument(
            "--bpe_tokenizer_path",
            type=str,
            default="src/dataloaders/data_representation/bpe/ecg_byte_tokenizer_10000.pkl",
            help="Please specify the path to the saved bpe tokenizer",
        )
        parser.add_argument("--num_workers", type=int, default=0, help="Please choose the num works for the dataloader")

        parser.add_argument(
            "--neural_network",
            type=str,
            default=None,
            help="Please choose the main neural network",
            choices=[
                "trans_discrete_decoder",
                "trans_discrete_encoder",
                "trans_discrete_seq2seq",
                "trans_continuous_nepa",
                "trans_continuous_dit",
                "mae_vit",
            ],
        )
        parser.add_argument(
            "--batch_labels",
            type=str,
            nargs="+",
            default = None,
            help="qrs_dur_wide, v_pacing, lbbb available for only batch datasets",
        )
        parser.add_argument("--add_task_head", action="store_true", default=None, help="Adds task head")
        parser.add_argument("--ckpt_has_head", action="store_true", default=None, help="indicates head is already trained")

        parser.add_argument("--cond_type", type=str, default=None, choices=["label", "text", "lead"],
                            help="Condition type for conditional diffusion (None = unconditional)")
        parser.add_argument("--cond_label_name", type=str, default=None, help="Which label key to condition on (e.g. v_pacing)")
        parser.add_argument("--cond_lead_index", type=int, default=1, help="Lead index for lead conditioning (0-indexed, default=1 for lead II)")
        parser.add_argument("--cond_drop_prob", type=float, default=0.1, help="Condition dropout probability for classifier-free guidance")
        parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-free guidance scale at inference (1.0 = no guidance)")
        parser.add_argument("--text_max_len", type=int, default=512, help="Max text length for text conditioning (byte-level)")

        parser.add_argument("--norm_eps", type=float, default=1e-6, help="Please choose the normalization epsilon")

        parser.add_argument("--wandb", action="store_true", default=None, help="Enable logging")

        parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
        parser.add_argument("--distributed", action="store_true", default=None, help="Enable distributed training")
        parser.add_argument("--nn_ckpt", type=str, default=None, help="Path to the NN checkpoint")
        parser.add_argument("--ema", action="store_true", default=None)
        parser.add_argument("--ema_decay", type=float, default=0.999)

    if "train" in mode:
        parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
        parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "muon"], help="Optimizer type")
        parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
        parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
        parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
        parser.add_argument("--patience_delta", type=float, default=0.1, help="Delta for early stopping")
        parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for optimizer")
        parser.add_argument("--beta2", type=float, default=0.95, help="Beta2 for optimizer")
        parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon for optimizer")
        parser.add_argument("--lr_schedule", type=str, default="constant", choices=["constant", "cosine", "inv_sqrt"], help="LR schedule after warmup")
        parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Min LR as fraction of peak LR (for cosine schedule)")
        parser.add_argument("--warmup", type=int, default=2000, help="Warmup steps")
        parser.add_argument("--ref_global_bs", type=int, default=None)
        parser.add_argument("--grad_accum_steps", type=int, default=1)
        parser.add_argument("--grad_clip", type=float, default=0.0, help="Max gradient norm for clipping (0 to disable)")
        parser.add_argument("--scale_wd", type=str, default="none", choices=["none", "inv_sqrt", "inv_linear"])
        parser.add_argument(
            "--torch_compile",
            action="store_true",
            default=None,
            help="Torch compile the model (should really only be used during pretraining or large finetuning.)",
        )
        parser.add_argument("--bfloat_16", action = "store_true", default = None)
    return parser.parse_args()