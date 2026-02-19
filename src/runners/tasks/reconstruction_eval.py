import torch
from tqdm import tqdm

from utils.gpu_setup import is_main
from utils.viz import plot_ecg
from utils.eval_stats import compute_reconstruction_metrics
from utils.runner_helpers import batch_to_device, stitch_12lead

from configs.constants import PTB_ORDER, PTB_INDEPENDENT_LEADS
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
    if args.condition == "lead":
        PTB_INDEPENDENT_LEADS.pop(args.condition_lead)
        lead_names = PTB_INDEPENDENT_LEADS
        n_leads = 7
    else:
        lead_names = PTB_ORDER
        n_leads = 12

    sums = {t: {m: torch.zeros(n_leads, device=device) for m in METRIC_NAMES} for t in t_values}
    n_samples = {t: 0 for t in t_values}
    num_viz = 30

    with torch.no_grad():
        for step, batch in enumerate(progress):
            report = batch["report"][0]
            batch = {k: batch_to_device(v, device) for k, v in batch.items()}
            
            if args.objective in ["ddpm", "rectified_flow"]:
                condition = batch["condition"] if args.condition else None
                for t in t_values:
                    out = nn.reconstruct(batch["signal"], t_value = t, condition = condition)
                    x1_pred = out["x1_pred"]

                    metrics = compute_reconstruction_metrics(x1_pred, batch["signal"], sf=args.sf)
                    for m in METRIC_NAMES:
                        sums[t][m] += metrics[m].sum(dim=0)
                    n_samples[t] += batch["signal"].shape[0]

                    if step < num_viz:
                        if args.condition == "lead":
                            x1_pred = stitch_12lead(args.condition_lead, condition, x1_pred)
                        plot_ecg(x1_pred[0].detach().cpu().numpy(), file_name = f"{step}_{t}_pred", 
                                 plot_title=report, save_dir=args.run_dir)
                        plot_ecg(out["xt"][0].detach().cpu().numpy(), leads = lead_names,
                                 file_name = f"{step}_{t}_noised", plot_title=report,
                                 save_dir=args.run_dir)
                if step < num_viz:
                    plot_ecg(batch["12_lead_gt"][0].detach().cpu().numpy(), 
                             file_name = f"{step}_gt", plot_title=report,
                             save_dir=args.run_dir)
                    
            if step > 590:
                break

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
            for i, lead in enumerate(PTB_ORDER):
                print(f"      {lead:>5}: MSE={per_lead['mse'][i]:.6f} | MAE={per_lead['mae'][i]:.6f} | PSNR={per_lead['psnr'][i]:.2f} | SSIM={per_lead['ssim'][i]:.4f}")
    return eval_dict