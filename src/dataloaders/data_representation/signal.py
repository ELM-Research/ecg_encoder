from typing import Tuple
import numpy as np

from configs.constants import BATCH_LABEL_CATS

class Signal:
    def __init__(self, args):
        self.args = args

    def __call__(self, data):
        normalized_data = self.normalize(data["ecg"])
        padded_data, padding_mask, modality_mask = self.pad(normalized_data)
        labels = {}
        if self.args.batch_labels:
            for name, arr in data.items():
                if name in BATCH_LABEL_CATS and name in self.args.batch_labels:
                    labels[name] = arr
        return {
            "transformed_data": padded_data,
            "padding_mask": padding_mask,
            "modality_mask": modality_mask,
            **labels,
        }

    def pad(self, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        M, N = signals.shape
        valid_len = min(N, self.args.segment_len)
        padded = np.zeros((12, self.args.segment_len), dtype=np.float32)
        padded[:M, :valid_len] = signals[:, :valid_len]

        padding_mask = np.arange(self.args.segment_len) >= valid_len
        modality_mask = np.arange(12) >= M

        return padded, padding_mask, modality_mask

    def normalize(self, arr: np.ndarray):
        mn = arr.min()
        mx = arr.max()
        denom = mx - mn + self.args.norm_eps
        normalized_arr = np.clip((arr - mn) / denom, 0.0, 1.0)
        return normalized_arr