import torch
import numpy as np
from tqdm import tqdm

from utils.gpu_setup import is_main
from utils.viz import plot_ecg
from utils.eval_stats import compute_fid, compute_mmd


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
    num_viz = 10

    with torch.no_grad():
        for step, batch in enumerate(progress):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            if "trans_continuous" in args.neural_network or "mae" in args.neural_network:
                signal = batch["signal"]
                bsz = signal.shape[0]
                condition = batch["condition"] if args.condition else None

                # Extract features from real data
                real_feat = nn.get_features(signal, batch.get("padding_mask"))
                real_features.append(real_feat.cpu().numpy())

                # Generate samples and extract features
                generated = nn.sample((bsz, 12, 2500), device=device,
                                      num_steps=nn.cfg.num_steps, condition=condition)
                gen_feat = nn.get_features(generated)
                gen_features.append(gen_feat.cpu().numpy())

                # Visualize a few samples
                if step < num_viz:
                    plot_ecg(generated[0].detach().cpu().numpy(), title=f"gen_{step}")

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
