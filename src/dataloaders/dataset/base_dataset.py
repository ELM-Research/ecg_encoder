from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os
import numpy as np
import random

from configs.constants import DATA_DIR

from utils.dir_file import DirFileManager

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
    if args.task in ["generation", "reconstruction", "forecasting"]:
        train, test = train_test_split(saved_dir, train_size = 0.7, random_state=args.seed)
    elif args.task == "pretrain":
        train = saved_dir
        test = saved_dir
    if "train" in args.mode:
        return train
    else:
        return test