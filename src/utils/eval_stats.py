import torch
import torch.nn.functional as F
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


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """Compute Fréchet Distance between two multivariate Gaussians N(mu1, sigma1) and N(mu2, sigma2).

    FD = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * sqrt(sigma1 @ sigma2))
    """
    from scipy.linalg import sqrtm

    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    # sqrtm can return complex values due to numerical error
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_fid(real_features: np.ndarray, gen_features: np.ndarray) -> float:
    """Compute FID between real and generated feature sets.

    Args:
        real_features: (N, D) array of features from real samples.
        gen_features: (M, D) array of features from generated samples.
    """
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
    return frechet_distance(mu1, sigma1, mu2, sigma2)


def compute_ssim_1d(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Compute SSIM for 1D signals per-lead.

    Args:
        pred: (B, C, T) predicted signals in [0, 1].
        target: (B, C, T) target signals in [0, 1].
        window_size: size of the 1D Gaussian window.

    Returns:
        (B, C) SSIM per sample per lead.
    """
    C = pred.shape[1]
    sigma = 1.5
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - window_size // 2
    gauss = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel = gauss.view(1, 1, -1).expand(C, -1, -1)
    pad = window_size // 2

    mu_x = F.conv1d(pred, kernel, padding=pad, groups=C)
    mu_y = F.conv1d(target, kernel, padding=pad, groups=C)
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv1d(pred ** 2, kernel, padding=pad, groups=C) - mu_x_sq
    sigma_y_sq = F.conv1d(target ** 2, kernel, padding=pad, groups=C) - mu_y_sq
    sigma_xy = F.conv1d(pred * target, kernel, padding=pad, groups=C) - mu_xy

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    )
    return ssim_map.mean(dim=2)  # (B, C)


def ssim_window_size(sf: int) -> int:
    """Derive an odd SSIM window size that spans ~44 ms (11 samples at 250 Hz)."""
    ws = int(11 * sf / 250)
    return ws | 1  # ensure odd


def compute_reconstruction_metrics(pred: torch.Tensor, target: torch.Tensor, sf: int = 250) -> dict[str, torch.Tensor]:
    """Compute per-lead reconstruction metrics between predicted and target signals.

    Args:
        pred: (B, C, T) predicted signals in [0, 1].
        target: (B, C, T) target signals in [0, 1].
        sf: sampling frequency in Hz (used to scale the SSIM window).

    Returns:
        dict with keys 'mse', 'mae', 'psnr', 'ssim', each of shape (B, C).
    """
    mse = (pred - target).pow(2).mean(dim=2)
    mae = (pred - target).abs().mean(dim=2)
    psnr = -10.0 * torch.log10(mse.clamp(min=1e-10))
    ssim = compute_ssim_1d(pred, target, window_size=ssim_window_size(sf))
    return {"mse": mse, "mae": mae, "psnr": psnr, "ssim": ssim}


def compute_mmd(real_features: np.ndarray, gen_features: np.ndarray, gamma: float | None = None) -> float:
    """Compute MMD with RBF kernel between real and generated feature sets.

    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    where k is the RBF kernel: k(a,b) = exp(-gamma * ||a-b||^2)

    Args:
        real_features: (N, D) array of features from real samples.
        gen_features: (M, D) array of features from generated samples.
        gamma: RBF bandwidth. If None, uses 1 / D (median heuristic alternative).
    """
    if gamma is None:
        gamma = 1.0 / real_features.shape[1]

    def rbf(x, y):
        # (N, 1, D) - (1, M, D) -> (N, M)
        dists_sq = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=-1)
        return np.exp(-gamma * dists_sq)

    xx = rbf(real_features, real_features).mean()
    yy = rbf(gen_features, gen_features).mean()
    xy = rbf(real_features, gen_features).mean()
    return float(xx + yy - 2 * xy)