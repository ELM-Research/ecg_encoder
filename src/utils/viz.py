import numpy as np
import matplotlib.pyplot as plt

from utils.dir_file import DirFileManager

from configs.constants import PTB_ORDER

def plot_ecg(ecg, leads = PTB_ORDER, sf = 250, file_name = None, plot_title = None, save_dir = None):
    n_leads, T = ecg.shape
    t = np.arange(T) / sf

    fig, axes = plt.subplots(n_leads, 1, figsize=(12, n_leads * 0.8), sharex = True)
    if n_leads == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(t, ecg[i], color = 'k', linewidth = 0.5)
        ax.set_ylabel(leads[i], fontsize=8, rotation=0, 
                      ha = "right", va = "center")
        # ax.set_ylim([0, 1])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
    
    axes[-1].set_xlabel("Time (s)")
    if plot_title:
        fig.suptitle(plot_title, fontsize=12)
    plt.tight_layout()
    DirFileManager.ensure_directory_exists(folder = f"{save_dir}/pngs")
    plt.savefig(f"{save_dir}/pngs/{file_name}.png", dpi = 150, bbox_inches = "tight")
    plt.close()

def plot_forecast(context, gt_future, pred_future, report, save_path):
    fig, ax = plt.subplots(figsize=(20, 4))
    n_ctx = len(context)
    fut_x = np.arange(n_ctx, n_ctx + max(len(gt_future), len(pred_future)))
    ax.plot(np.arange(n_ctx), context, color="black", linewidth=1, label="Context")
    ax.plot(fut_x[: len(gt_future)], gt_future, color="silver", linewidth=1, label="Ground Truth")
    ax.plot(fut_x[: len(pred_future)], pred_future, color="tab:red", linewidth=1, label="Prediction")
    ax.axvline(n_ctx, color="gray", linestyle="--", linewidth=0.5)
    ax.set_title(report, fontsize=14)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close()
