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
        result = {
            "transformed_data": padded_data,
            "padding_mask": padding_mask,
            "modality_mask": modality_mask,
            **labels,
        }

        cond_type = getattr(self.args, "cond_type", None)
        if cond_type == "label":
            label_name = self.args.cond_label_name
            result["cond"] = np.int64(data[label_name])
        elif cond_type == "text":
            report = data.get("report", "")
            max_len = getattr(self.args, "text_max_len", 512)
            result["cond"] = self.encode_text(report, max_len)
        elif cond_type == "lead":
            result["cond"] = padded_data[self.args.cond_lead_index].astype(np.float32)

        return result

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

    @staticmethod
    def encode_text(text, max_len: int = 512) -> np.ndarray:
        if isinstance(text, (bytes, np.bytes_)):
            text = text.decode("utf-8", errors="replace")
        text = str(text)
        encoded = [b + 1 for b in text.encode("utf-8")[:max_len]]
        padded = encoded + [0] * (max_len - len(encoded))
        return np.array(padded, dtype=np.int64)
