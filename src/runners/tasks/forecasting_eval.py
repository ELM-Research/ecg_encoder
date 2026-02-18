import torch
from tqdm import tqdm
import numpy as np

from utils.gpu_setup import is_main
from utils.viz import plot_forecast
from utils.eval_stats import forecast_metrics
from utils.runner_helpers import batch_to_device
from utils.dir_file import DirFileManager

def eval_forecasting(nn, dataloader, args):
    show_progress = is_main()
    nn.eval()
    progress = tqdm(
        dataloader,
        desc="Evaluating Forecasting",
        disable=not show_progress,
        leave=False,
    )
    device = next(nn.parameters()).device
    data_repr = dataloader.dataset.data_representation
    plot_dir = f"{args.run_dir}/forecast_plots"
    DirFileManager.ensure_directory_exists(folder=plot_dir)
    all_acc, all_sig = [], []

    with torch.no_grad():
        for step, batch in enumerate(progress):
            report = batch["report"][0]
            labels = batch["labels"].numpy()
            batch = {k: batch_to_device(v, device) for k, v in batch.items()}
            pred = nn.generate(batch["tgt_ids"].to(device), max_new_tokens=labels.shape[1],
                               return_new_only=True, return_logits = False).cpu().numpy()
            gt = labels[:, :pred.shape[1]]
            all_acc.append((pred == gt).mean())
            for i in range(len(gt)):
                mn, mx = batch["min"][i].item(), batch["max"][i].item()
                ps = data_repr.decode(pred[i].tolist(), mn, mx)
                gs = data_repr.decode(gt[i].tolist(), mn, mx)
                all_sig.append(forecast_metrics(ps, gs))
                if step < 30 and i == 0:
                    tgt_tokens = batch["tgt_ids"][0].tolist()
                    ctx_signal = data_repr.decode(tgt_tokens, mn, mx)
                    plot_forecast(ctx_signal, gs, ps, report, f"{plot_dir}/plot_{step}.png")
            if step > 1000:
                break

    metrics = {"accuracy": float(np.mean(all_acc))}
    for k in all_sig[0]:
        metrics[k] = float(np.mean([s[k] for s in all_sig]))
    print("Forecast | " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics
