import torch
from configs.constants import PTB_INDEPENDENT_IDX

def batch_to_device(v, device):
    if isinstance(v, torch.Tensor):
        return v.to(device)
    if isinstance(v, dict):
        return {k: batch_to_device(x, device) for k, x in v.items()}
    return v

def stitch_12lead(cond_idx, x_cond, x_gen):
    other = [i for i in PTB_INDEPENDENT_IDX if i != cond_idx]
    ecg = torch.zeros(1, 12, 2500, device=x_gen.device)
    ecg[:, cond_idx] = x_cond
    for i, idx in enumerate(other):
        ecg[:, idx] = x_gen[:, i]
    I, II = ecg[:, 0], ecg[:, 1]
    ecg[:, 2] = II - I
    ecg[:, 3] = -0.5 * (I + II)
    ecg[:, 4] = 0.5 * (I - ecg[:, 2])
    ecg[:, 5] = 0.5 * (II + ecg[:, 2])
    return ecg