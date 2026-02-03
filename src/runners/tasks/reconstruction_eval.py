import torch
from tqdm import tqdm

from configs.constants import PTB_ORDER
from utils.gpu_setup import is_main
from utils.viz import plot_ecg
from utils.eval_stats import compute_reconstruction_metrics

METRIC_NAMES = ["mse", "mae", "psnr", "ssim"]

def eval_reconstruction(nn, dataloader, args):
    show_progress = is_main()
    nn.eval()
    progress = tqdm(
        dataloader,
        desc="Evaluating Reconstruction",
        disable=not show_progress,
        leave=False,
    )
    device = next(nn.parameters()).device
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_leads = len(PTB_ORDER)

    # Accumulators per t-value
    sums = {t: {m: torch.zeros(n_leads, device=device) for m in METRIC_NAMES} for t in t_values}
    n_samples = {t: 0 for t in t_values}

    with torch.no_grad():
        for step, batch in enumerate(progress):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            signal = batch["signal"]  # (B, C, T)

            if "trans_continuous" in args.neural_network or "mae" in args.neural_network:
                for t in t_values:
                    out = nn.reconstruct(signal, t_value=t)
                    x1_pred = out["x1_pred"]  # (B, C, T)

                    metrics = compute_reconstruction_metrics(x1_pred, signal)
                    for m in METRIC_NAMES:
                        sums[t][m] += metrics[m].sum(dim=0)  # (C,)
                    n_samples[t] += signal.shape[0]

                    if step <= 10:
                        plot_ecg(x1_pred[0].detach().cpu().numpy(), title=f"{step}_{t}_pred")
                        plot_ecg(out["xt"][0].detach().cpu().numpy(), title=f"{step}_{t}_noised")

    # Build eval dict and print results
    eval_dict = {}
    if show_progress:
        print("\nReconstruction metrics:")

    for t in t_values:
        if n_samples[t] == 0:
            continue
        per_lead = {m: (sums[t][m] / n_samples[t]).cpu() for m in METRIC_NAMES}
        t_key = str(t)

        for m in METRIC_NAMES:
            eval_dict[f"{m}/t={t_key}/global"] = per_lead[m].mean().item()
            for i, lead in enumerate(PTB_ORDER):
                eval_dict[f"{m}/t={t_key}/{lead}"] = per_lead[m][i].item()

        if show_progress:
            g = {m: per_lead[m].mean().item() for m in METRIC_NAMES}
            print(f"  t={t} — MSE: {g['mse']:.6f} | MAE: {g['mae']:.6f} | PSNR: {g['psnr']:.2f} dB | SSIM: {g['ssim']:.4f}")

    return eval_dict
