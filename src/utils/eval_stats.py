import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    return (pred == target).mean()

def roc_auc(logits, labels, save_dir):
    auroc = roc_auc_score(labels, logits)
    fpr, tpr, thresholds = roc_curve(labels, logits)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
    plt.plot([0,1], [0,1], "k--", label = "Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(save_dir)
    return auroc

def f1(pred: np.ndarray, target: np.ndarray) -> float:
    return f1_score(target.flatten(), pred.flatten(), average="macro", zero_division=0)


def aggregate_metrics(all_metrics):
    keys = all_metrics[0].keys()
    results = {}
    for k in keys:
        values = [m[k] for m in all_metrics]
        results[k] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}
    return results