import torch
from tqdm import tqdm
from utils.gpu_setup import is_main
from utils.eval_stats import accuracy, f1, roc_auc
from utils.runner_helpers import batch_to_device

from configs.constants import BATCH_LABEL_CATS

def eval_classification(nn, dataloader, args):
    show_progress = is_main()
    nn.eval()
    progress = tqdm(
        dataloader,
        desc="Evaluating Classification",
        disable=not show_progress,
        leave=False,
    )
    device = next(nn.parameters()).device
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in progress:
            batch = {k: batch_to_device(v, device) for k, v in batch.items()}
            
            if "trans_discrete" in args.neural_network:
                out, logits = nn.generate(
                    tgt_ids=batch["tgt_ids"],
                    max_new_tokens=3,
                    return_new_only=True,
                    return_logits=True,
                )
                label_name = args.batch_labels[-1]
                class_ids = [getattr(args, f"{label_name}_{int(v)}") for v in BATCH_LABEL_CATS[label_name]]
                class_out = out[:, 1::2][:, -1]
                class_logits = logits[:, 1::2][:, -1, class_ids]
                class_labels = batch["labels"][:, 1::2][:, -1]
                probs = torch.softmax(class_logits, dim=-1)[:, 1]
                preds = (class_out == class_ids[1]).long()
                labels = (class_labels == class_ids[1]).long()
            elif "trans_continuous" in args.neural_network or "mae" in args.neural_network:
                out = nn(**batch)
                if args.task == "multiclass_classification":
                    label_name = args.batch_labels[-1]
                    logits = out.logits[label_name]
                    probs = torch.softmax(logits, dim=-1)[:, 1]
                    preds = logits.argmax(dim=-1)
                    labels = batch[label_name]
                elif args.task == "multilabel_classification":
                    logits = out.logits
                    preds = (logits > 0).int()
                    targets = torch.stack([batch[k] for k in args.batch_labels], dim=-1)
                    

            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds).float().numpy()
    all_probs = torch.cat(all_probs).float().numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    if args.task == "multilabel_classification":
        results = {}
        for i, label_name in enumerate(args.batch_labels):
            p, pr, l = all_preds[:, i], all_probs[:, i], all_labels[:, i]
            acc = accuracy(p, l)
            f1_score = f1(p, l)
            save_dir = f"{args.run_dir}/roc_auc_curve_{label_name}_{('_').join(args.data)}.png"
            auroc = roc_auc(pr, l, save_dir)
            print(f"{label_name} — ROC AUC: {auroc:.4f} | Acc: {acc:.4f} | F1: {f1_score:.4f}")
            results[label_name] = {"accuracy": acc, "f1": f1_score, "roc_auc": auroc}
        return results
    else:
        label_name = args.batch_labels[-1]
        acc = accuracy(all_preds, all_labels)
        f1_score = f1(all_preds, all_labels)
        save_dir = f"{args.run_dir}/roc_auc_curve_{label_name}_{('_').join(args.data)}.png"
        auroc = roc_auc(all_probs, all_labels, save_dir)
        print(f"ROC AUC: {auroc:.4f} | Acc: {acc:.4f} | F1: {f1_score:.4f}")
        return {"accuracy": acc, "f1": f1_score, "roc_auc": auroc}
    
    print(f"ROC AUC: {auroc:.4f} | Acc: {acc:.4f} | F1: {f1_score:.4f}")
    return {"accuracy": acc, "f1": f1_score, "roc_auc": auroc}
