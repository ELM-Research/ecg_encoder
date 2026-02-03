import torch
from tqdm import tqdm

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

    with torch.no_grad():
        for step, batch in enumerate(progress):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            if "trans_continuous" in args.neural_network or "mae" in args.neural_network:
                for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    out = nn.reconstruct(batch["signal"], t_value = t)
                    plot_ecg(out["x1_pred"][0].detach().cpu().numpy(), title = f"{step}_{t}_pred")
                    plot_ecg(out["xt"][0].detach().cpu().numpy(), title = f"{step}_{t}_noised")
                    
            if step > 10:
                break
