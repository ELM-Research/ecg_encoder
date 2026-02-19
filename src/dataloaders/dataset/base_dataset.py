from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os
import numpy as np
import random
from collections import Counter

from configs.constants import DATA_DIR

from utils.dir_file import DirFileManager
from utils.gpu_setup import is_main

class BaseDataset(Dataset):
    def __init__(self, data, data_representation, task, args):
        self.data = data
        self.args = args
        self.data_representation = data_representation
        self.task = task
        self.dfm = DirFileManager()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        npy_file = self.dfm.open_npy(self.data[index])
        if self.args.augment:
            npy_file["ecg"] = self.augment_ecg(npy_file["ecg"])
        
        transformed_data = self.data_representation(npy_file)
        out = self.task(transformed_data)
        return out
    
    def augment_ecg(self, signal):
        if random.random() < 0.25:
            noise_level = 0.05
            noise = np.random.normal(0, noise_level * np.std(signal), signal.shape)
            perturbed_signal = signal + noise

            if random.random() < 0.25:
                wander_amplitude = 0.07 * np.max(np.abs(signal))
                wander = wander_amplitude * np.sin(np.linspace(0, random.randint(1, 5) * np.pi, signal.shape[1]))
                wander = np.tile(wander, (signal.shape[0], 1))
                perturbed_signal += wander

            return perturbed_signal
        return signal
    
def load_base_dataset(data, args):
    saved_path = f"{DATA_DIR}/{data}/preprocessed_{args.segment_len}"
    saved_dir = sorted([os.path.join(saved_path, f) for f in os.listdir(saved_path)])
    if "classification" in args.task:
        if is_main():
            print("Filtering dataset...")
        if "batch" in data:
            if args.task == "multilabel_classification":
                if data == "batch13":
                    batch_labels = ["lbbb"]
                elif data == "batch12":
                    batch_labels = ["v_pacing"]
                elif data == "batch14":
                    batch_labels = ["qrs_dur_wide"]
            elif args.task == "multiclass_classification":
                batch_labels = [args.batch_labels[-1]]

            train, test = filter_dataset(saved_dir, batch_labels, seed = args.seed,)
    elif args.task in ["generation", "reconstruction", "forecasting"]:
        train, test = train_test_split(saved_dir, train_size = 0.7, random_state=args.seed)
    elif args.task == "pretrain":
        train = saved_dir
        test = saved_dir
    if "train" in args.mode:
        return train
    else:
        return test

def filter_dataset(paths, labels, train_ratio=0.7, seed=42):
    rng = np.random.default_rng(seed)
    cache = {path: DirFileManager.open_npy(path) for path in paths}
    
    remaining = set(paths)
    for label in labels:
        groups = {}
        for path in remaining:
            cls = cache[path][label]
            groups.setdefault(cls, []).append(path)
        
        min_count = min(len(p) for p in groups.values())
        if is_main():
            print(f"Original: {label} = {{{', '.join(f'{k}: {len(v)}' for k, v in groups.items())}}}")
            print(f"Truncating to {min_count} per class")
        
        balanced = []
        for cls_paths in groups.values():
            balanced.extend(rng.choice(cls_paths, min_count, replace=False))
        remaining = set(balanced)
    
    remaining = list(remaining)
    rng.shuffle(remaining)
    n_train = int(len(remaining) * train_ratio)
    train, test = remaining[:n_train], remaining[n_train:]
    # n_train = int(4974/2)
    # n_test = int(2132/2)
    # train, test = remaining[:n_train], remaining[-n_test:]
    
    if is_main():
        for label in labels:
            train_dist = Counter(cache[p][label] for p in train)
            test_dist = Counter(cache[p][label] for p in test)
            print(f"Final {label}: train={dict(train_dist)}, test={dict(test_dist)}")
        print(f"Total: train={len(train)}, test={len(test)}")
    return train, test

# '''
# Original: v_pacing = {0: 214598, 1: 3553}
# Truncating to 3553 per class
# Original: qrs_dur_wide = {1: 4022, 0: 3084}
# Truncating to 3084 per class
# Final v_pacing: train={0: 2413, 1: 1904}, test={0: 1032, 1: 819}
# Final qrs_dur_wide: train={0: 2159, 1: 2158}, test={0: 925, 1: 926}
# Total: train=4317, test=1851
# Filtering dataset...
# Original: v_pacing = {0: 16118, 1: 253}
# Truncating to 253 per class
# Original: qrs_dur_wide = {1: 295, 0: 211}
# Truncating to 211 per class
# Final v_pacing: train={0: 168, 1: 127}, test={0: 76, 1: 51}
# Final qrs_dur_wide: train={0: 147, 1: 148}, test={0: 64, 1: 63}
# Total: train=295, test=127
# Length of Dataset: 4612

# Original: qrs_dur_wide = {0: 186390, 1: 31761}
# Truncating to 31761 per class
# Original: v_pacing = {0: 59981, 1: 3541}
# Truncating to 3541 per class
# Final qrs_dur_wide: train={1: 3617, 0: 1340}, test={1: 1602, 0: 523}
# Final v_pacing: train={0: 2502, 1: 2455}, test={1: 1086, 0: 1039}
# Total: train=4957, test=2125
# Filtering dataset...
# Original: qrs_dur_wide = {0: 14107, 1: 2264}
# Truncating to 2264 per class
# Original: v_pacing = {0: 4276, 1: 252}
# Truncating to 252 per class
# Final qrs_dur_wide: train={1: 261, 0: 91}, test={0: 46, 1: 106}
# Final v_pacing: train={0: 168, 1: 184}, test={0: 84, 1: 68}
# Total: train=352, test=152
# Length of Dataset: 5309


# Original: qrs_dur_wide = {0: 186390, 1: 31761}
# Truncating to 31761 per class
# Final qrs_dur_wide: train={1: 22296, 0: 22169}, test={0: 9592, 1: 9465}
# Total: train=44465, test=19057
# Filtering dataset...
# Original: qrs_dur_wide = {0: 14107, 1: 2264}
# Truncating to 2264 per class
# Final qrs_dur_wide: train={0: 1592, 1: 1577}, test={0: 672, 1: 687}
# Total: train=3169, test=1359
# Length of Dataset: 47634



# Original: v_pacing = {0: 214598, 1: 3553}
# Truncating to 3553 per class
# Final v_pacing: train={0: 2515, 1: 2459}, test={0: 1038, 1: 1094}
# Total: train=4974, test=2132
# Filtering dataset...
# Original: v_pacing = {0: 16118, 1: 253}
# Truncating to 253 per class
# Final v_pacing: train={0: 176, 1: 178}, test={1: 75, 0: 77}
# Total: train=354, test=152
# Length of Dataset: 5328
# '''

# # Final v_pacing: train={0: 2515, 1: 2459}, test={0: 1038, 1: 1094}