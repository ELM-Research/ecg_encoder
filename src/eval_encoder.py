import gc
import torch
import json

from configs.config import get_args

from utils.gpu_setup import GPUSetup, is_main
from utils.seed_setup import set_seed
from utils.eval_stats import aggregate_metrics
from utils.wandb_setup import setup_wandb, cleanup_wandb, log_wandb

from neural_networks.build_nn import BuildNN

from dataloaders.build_dataloader import BuildDataLoader
from runners.tasks.classification_eval import eval_classification
from runners.tasks.generation_eval import eval_generation
from runners.tasks.reconstruction_eval import eval_reconstruction
from runners.tasks.forecasting_eval import eval_forecasting

TASK_RUNNERS = {
    "multilabel_classification": eval_classification,
    "multiclass_classification": eval_classification,
    "generation": eval_generation,
    "reconstruction": eval_reconstruction,
    "forecasting": eval_forecasting,
}


def main():
    gc.collect()
    torch.cuda.empty_cache()
    mode = "downstream_eval"
    args = get_args(mode)
    args.mode = mode
    assert args.task is not None, print("Please specify a task")

    if args.nn_ckpt:
        data_names = "_".join(args.data)
        batch_label_names = "_".join(args.batch_labels) if args.batch_labels else "None"
        run_dir = "/".join(args.nn_ckpt.split("/")[:-2])
        args.run_dir = run_dir
    
    if is_main():
        print(f"Evaluating {args.task}")
        if args.wandb:
            name = f"{args.objective}_{batch_label_names}_{data_names}"
            setup_wandb(args, name)

    eval_fn = TASK_RUNNERS[args.task]

    # seeds = [0, 1, 2, 3, 4]
    seeds = [0, 1]
    all_metrics = []
    for seed in seeds:
        args.seed = seed
        print(f"Inferencing {args.seed}")
        set_seed(args.seed)
        build_dataloader = BuildDataLoader(args)
        dataloader = build_dataloader.build_dataloader()
        build_nn = BuildNN(args)
        nn_components = build_nn.build_nn(dataloader.dataset.data_representation)
        gpu_setup = GPUSetup(args)
        nn = gpu_setup.setup_gpu(nn_components["neural_network"], nn_components["find_unused_parameters"])
        if args.dev:
            gpu_setup.print_model_device(nn, f"{args.neural_network}")

        eval_result = eval_fn(nn, dataloader, args)
        all_metrics.append(eval_result)
        if is_main() and args.wandb:
            log_wandb(eval_result, prefix = f"seed_{seed}")

    results = aggregate_metrics(all_metrics)
    if is_main() and args.wandb:
        log_wandb(results, prefix = "agg")
        cleanup_wandb()
    if args.nn_ckpt:
        with open(f"{run_dir}/{args.task}_{batch_label_names}_{data_names}_results.json", "w") as f:
            json.dump(results, f, indent=2)
    else:
        print(results)


if __name__ == "__main__":
    main()
