import os
import random
import tifffile
import cv2
import numpy as np
import logging

from DLIP.data.base_classes.base_pl_datamodule import BasePLDataModule
from DLIP.data.base_classes.segmentation.base_seg_dataset import BaseSegmentationDataset

class GenericSegmentationDataModule(BasePLDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size = 1,
        dataset_size = 1.0,
        val_to_train_ratio = 0,
        initial_labeled_ratio= 1.0,
        train_transforms=None,
        train_transforms_unlabeled=None,
        val_transforms=None,
        test_transforms=None,
        return_unlabeled_trafos=False,
        num_workers=0,
        pin_memory=False,
        shuffle=True,
        drop_last=False,
        binarize_labels = False,
    ):
        super().__init__(
            dataset_size=dataset_size,
            batch_size = batch_size,
            val_to_train_ratio = val_to_train_ratio,
            num_workers = num_workers,
            pin_memory = pin_memory,
            shuffle = shuffle,
            drop_last = drop_last,
            initial_labeled_ratio = initial_labeled_ratio,
        )
        self.root_dir = root_dir
        self.train_labeled_root_dir     = os.path.join(self.root_dir, "train")
        self.train_unlabeled_root_dir   = os.path.join(self.root_dir, "unlabeled")
        self.test_labeled_root_dir      = os.path.join(self.root_dir, "test")
        self.train_transforms = train_transforms
        self.train_transforms_unlabeled = (
            train_transforms_unlabeled
            if train_transforms_unlabeled is not None
            else train_transforms
        )
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.return_unlabeled_trafos = return_unlabeled_trafos
        self.labeled_train_dataset: BaseSegmentationDataset = None
        self.unlabeled_train_dataset: BaseSegmentationDataset = None
        self.val_dataset: BaseSegmentationDataset = None
        self.test_dataset: BaseSegmentationDataset = None
        self.binarize_labels = binarize_labels
        self.__init_datasets()

    def __init_datasets(self):
        self.labeled_train_dataset = BaseSegmentationDataset(
            root_dir=self.train_labeled_root_dir, 
            transforms=self.train_transforms,
        )
        for _ in range(int(len(self.labeled_train_dataset) * (1 - self.dataset_size))):
            self.labeled_train_dataset.pop_sample(random.randrange(len(self.labeled_train_dataset)))
            
        
        if os.path.exists(os.path.join(self.root_dir,'val')):
            self.val_dataset = BaseSegmentationDataset(
                root_dir=os.path.join(self.root_dir,'val'), 
                transforms=self.val_transforms,
            )
        else:
            self.val_dataset = BaseSegmentationDataset(
                root_dir=self.train_labeled_root_dir, 
                transforms=self.val_transforms,
            )
        
        self.test_dataset = BaseSegmentationDataset(
            root_dir=self.test_labeled_root_dir, 
            transforms=self.test_transforms,
        )


    def init_val_dataset(self, split_lst=None):
        return