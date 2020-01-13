import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
from skimage import io

class MelSpecDataset(Dataset):
    """Dataset of MFCC spectrograms for various music genres."""
    def __init__(self, csv_loc, data_dir, transforms=None):
        # dataframe with columns filename, genre, relative_dir (to data folder)
        self.data_list = pd.read_csv(csv_loc, index_col=0)
        self.data_dir = data_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """Used so that dataset[i] can be used to get the ith item."""

        if torch.istensor(idx):
            idx = idx.tolist()

        spec_path = os.path.join(self.data_dir, self.data_list.loc[idx, 'relative_dir'])
        spec = io.imread(spec_path)

        label = self.data_list.loc[idx, 'genre']

        sample = {'image': image, 'label': label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample