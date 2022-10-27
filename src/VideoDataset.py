"""PyTorch Dataset for video segmentation as a sequence of disconnected image frames

"""
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset

from typing import Callable, Optional, Sequence


class VideoDataset(Dataset):
    """Creates a list of (image, mask, weight) triplets for training a
    neural net on a semantic segmentation task.

    The weight arrays are generated from the masks. Data augmentation
    may be specified.

    """
    def __init__(
            self,
            data_dir: str,
            transform: Optional[Callable] = None):
        """Constructor.

        Args:
            data_dir (str): Directory containing a dataset's image, mask, and
                weight data. Assumes a directory structure of
                    data_dir/
                        image/
                        label/
                        class_weights.json
                where class_weights.json contains a list `weights` such that the
                correct weight for a pixel of class i is `weights[i]`.
            transform (Optional[Callable]): Optional data augmentation transform
                to be applied to each sample.
        """
        self.transform = transform
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'image')
        self.mask_dir = os.path.join(data_dir, 'label')
        with open(os.path.join(data_dir, 'class_weights.json'), 'r') as fd:
            self.class_weights = np.array(json.load(fd))

        # Get the filenames, under the assumption that corresponding images
        # and masks have the same filename with different extensions
        self.vid_names = os.listdir(self.mask_dir)
        self.filenames = [
            os.path.join(vdir, f)
            for vdir in self.vid_names
            for f in os.listdir(os.path.join(self.mask_dir, vdir))
            if os.path.splitext(f)[1] == '.png']
        return

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_file = os.path.join(self.image_dir, filename)
        mask_file = os.path.join(self.mask_dir, filename)

        image = plt.imread(image_file)[..., 0]
        mask_bw = plt.imread(mask_file)[..., :3].max(axis=2)
        if mask_bw.max() > 0:
            mask = (mask_bw == mask_bw.max()).astype(int)
        else: 
            mask = mask_bw.astype(int)

        shape = image.shape
        fixed_shape = (shape[0] // 2, shape[1] // 2)
        image = cv2.resize(image, fixed_shape, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, fixed_shape, interpolation = cv2.INTER_NEAREST)

        weight = self.class_weights[mask]

        sample = (image, mask, weight)

        if self.transform:
            sample = self.transform(sample)

        return sample

