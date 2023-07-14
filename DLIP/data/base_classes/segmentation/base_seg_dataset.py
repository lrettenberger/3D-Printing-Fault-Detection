from typing import Callable
import tifffile
import glob
import os
import numpy as np
import cv2
import random
import torch

from DLIP.data.base_classes.base_dataset import BaseDataset


class BaseSegmentationDataset(BaseDataset):
    def __init__(
        self,
        root_dir: str,
        samples_dir: str = "samples",
        samples_data_format="png",
        transforms = None,
        empty_dataset=False,
    ):

        self.empty_dataset = empty_dataset
        self.root_dir = root_dir
        self.samples_dir = samples_dir
        self.samples_data_format = samples_data_format
        self.transforms = transforms
        self.internal_seed = 0

        if transforms is None:
                self.transforms = lambda x, y: (x,y,0)
        if isinstance(transforms, list):
            self.transforms = transforms
        else:
            self.transforms = [self.transforms]

        self.samples = glob.glob(f"{os.path.join(self.root_dir,'0')}/*") + glob.glob(f"{os.path.join(self.root_dir,'1')}/*") + glob.glob(f"{os.path.join(self.root_dir,'2')}/*")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # load sample
        sample_img = cv2.imread(self.samples[idx],-1)
        label = int(self.samples[idx].split('/')[-2])
        return torch.tensor(sample_img / 255.).to(torch.float32).permute(2,0,1), label



    def pop_sample(self, index):
        return self.indices.pop(index)

    def add_sample(self, new_sample):
        self.indices.append(new_sample)

    def get_samples(self):
        return self.indices
