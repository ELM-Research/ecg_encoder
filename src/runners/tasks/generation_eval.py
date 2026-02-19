import torch
from tqdm import tqdm
import numpy as np

from utils.gpu_setup import is_main
from utils.viz import plot_ecg
from utils.eval_stats import compute_fid, compute_mmd
from utils.runner_helpers import batch_to_device, stitch_12lead

def eval_generation(nn, dataloader, args):
    show_progress = is_main()
    nn.eval()
    progress = tqdm(
        dataloader,
        desc="Evaluating Generation",
        disable=not show_progress,
        leave=False,
    )
    device = next(nn.parameters()).device
    real_features = []
    gen_features = []
    num_viz = 30
    num_leads = 7 if args.condition == "lead" else 12

    with torch.no_grad():
        for step, batch in enumerate(progress):
            report = batch["report"][0]
            batch = {k: batch_to_device(v, device) for k, v in batch.items()}
            
            if args.objective in ["ddpm", "rectified_flow"]:
                condition = batch["condition"] if args.condition else None
                bsz = batch["signal"].shape[0]
                real_feat = nn.get_features(batch["signal"], batch.get("padding_mask"))
                real_features.append(real_feat.cpu().numpy())
                generated = nn.sample((bsz, num_leads, 2500), device=device,
                                      num_steps=nn.cfg.num_steps, condition=condition)
                gen_feat = nn.get_features(generated)
                gen_features.append(gen_feat.cpu().numpy())
                if step < num_viz:
                    if args.condition == "lead":
                        generated = stitch_12lead(args.condition_lead, condition, generated)
                    plot_ecg(generated[0].detach().cpu().numpy(), file_name = step, 
                             plot_title = report, save_dir=args.run_dir)

            if step > 590:
                break


    if not real_features:
        return {}

    real_features = np.concatenate(real_features, axis=0)
    gen_features = np.concatenate(gen_features, axis=0)

    fid = compute_fid(real_features, gen_features)
    mmd = compute_mmd(real_features, gen_features)

    metrics = {"fid": fid, "mmd": mmd}

    if show_progress:
        print(f"Generation metrics — FID: {fid:.4f} | MMD: {mmd:.6f}")

    return metrics