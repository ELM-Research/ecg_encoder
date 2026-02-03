import torch
from tqdm import tqdm

from configs.constants import PTB_ORDER
from utils.gpu_setup import is_main
from utils.viz import plot_ecg

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

    # Accumulators per t-value: sum of errors and sample count
    mse_sum = {t: torch.zeros(n_leads, device=device) for t in t_values}
    mae_sum = {t: torch.zeros(n_leads, device=device) for t in t_values}
    psnr_sum = {t: torch.zeros(n_leads, device=device) for t in t_values}
    n_samples = {t: 0 for t in t_values}

    with torch.no_grad():
        for step, batch in enumerate(progress):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            signal = batch["signal"]  # (B, C, T)

            if "trans_continuous" in args.neural_network or "mae" in args.neural_network:
                for t in t_values:
                    out = nn.reconstruct(signal, t_value=t)
                    x1_pred = out["x1_pred"]  # (B, C, T)

                    # Per-lead MSE/MAE, averaged over time then summed over batch
                    per_sample_mse = (x1_pred - signal).pow(2).mean(dim=2)  # (B, C)
                    ae = (x1_pred - signal).abs().mean(dim=2)               # (B, C)
                    mse_sum[t] += per_sample_mse.sum(dim=0)  # (C,)
                    mae_sum[t] += ae.sum(dim=0)              # (C,)

                    # PSNR per-sample per-lead, then sum (signals are [0,1] so MAX=1)
                    psnr = -10.0 * torch.log10(per_sample_mse.clamp(min=1e-10))  # (B, C)
                    psnr_sum[t] += psnr.sum(dim=0)  # (C,)

                    n_samples[t] += signal.shape[0]

                    if step <= 10:
                        plot_ecg(x1_pred[0].detach().cpu().numpy(), title=f"{step}_{t}_pred")
                        plot_ecg(out["xt"][0].detach().cpu().numpy(), title=f"{step}_{t}_noised")

    # Build eval dict
    eval_dict = {}
    for t in t_values:
        if n_samples[t] == 0:
            continue
        per_lead_mse = (mse_sum[t] / n_samples[t]).cpu()    # (C,)
        per_lead_mae = (mae_sum[t] / n_samples[t]).cpu()    # (C,)
        per_lead_psnr = (psnr_sum[t] / n_samples[t]).cpu()  # (C,)

        t_key = str(t)
        eval_dict[f"mse/t={t_key}/global"] = per_lead_mse.mean().item()
        eval_dict[f"mae/t={t_key}/global"] = per_lead_mae.mean().item()
        eval_dict[f"psnr/t={t_key}/global"] = per_lead_psnr.mean().item()
        for i, lead in enumerate(PTB_ORDER):
            eval_dict[f"mse/t={t_key}/{lead}"] = per_lead_mse[i].item()
            eval_dict[f"mae/t={t_key}/{lead}"] = per_lead_mae[i].item()
            eval_dict[f"psnr/t={t_key}/{lead}"] = per_lead_psnr[i].item()

    return eval_dict
