import torch
from tqdm import tqdm

from utils.gpu_setup import is_main
from utils.viz import plot_ecg

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

    with torch.no_grad():
        for step, batch in enumerate(progress):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            if "trans_continuous" in args.neural_network or "mae" in args.neural_network:
                condition = batch["condition"] if args.condition else None
                out = nn.sample((1, 12, 2500), device = device, 
                                num_steps = nn.cfg.num_steps, condition = condition,)
                                # cfg_scale = 5)
                plot_ecg(out[0].detach().cpu().numpy(), title = step)

            if step > 10:
                break
