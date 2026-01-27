import torch
from tqdm import tqdm
import numpy as np

from utils.gpu_setup import is_main
from utils.eval_stats import accuracy, f1, roc_auc


def eval_classification(nn, dataloader, args):
    show_progress = is_main()
    nn.eval()
    progress = tqdm(
        dataloader,
        desc="Evaluating Classification",
        disable=not show_progress,
        leave=False,
    )
    num_labels = 3
    device = next(nn.parameters()).device
    all_acc, all_f1 = [], []
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for step, batch in enumerate(progress):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if "trans_discrete" in args.neural_network:
                out = nn.generate(
                    tgt_ids=batch["tgt_ids"],
                    max_new_tokens=num_labels,
                    return_new_only=True,
                )
                labels = batch["labels"].detach().cpu().numpy()
            elif "trans_continuous" in args.neural_network or \
                "mae" in args.neural_network:
                out = nn(**batch)
                for label in args.batch_labels:
                    logits = out.logits[label]
                    all_logits.append(logits[:, 1].detach().cpu())
                    out = logits.argmax(dim=1)
                    labels = batch[label].detach().cpu().numpy()
                    all_labels.append(labels)
            
            pred = out.detach().cpu().numpy()
            if pred.shape == labels.shape:
                all_acc.append(accuracy(pred, labels))
                all_f1.append(f1(pred, labels))
            else:
                all_acc.append(0)
                all_f1.append(0)

    if len(all_logits) != 0:
        all_logits = torch.cat(all_logits).numpy()
        all_labels = np.array(all_labels)
        data_names = "_".join(args.data)
        batch_label_names = "_".join(args.batch_labels)
        auroc = roc_auc(all_logits, all_labels, f"{args.run_dir}/roc_auc_curve_{batch_label_names}_{data_names}.png")
        print(f"ROC AUC: {auroc}")
    print(f"Acc: {np.mean(all_acc):.4f} | F1: {np.mean(all_f1):.4f}")
    return {"accuracy": float(np.mean(all_acc)), "f1": float(np.mean(all_f1)),
            "roc_auc": auroc}