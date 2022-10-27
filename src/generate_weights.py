"""

"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def generate_weights(
        data_dir: str,
        n_classes: int):
    """Compute class frequency-balancing weights from the masks saved in
    a directory, and save them to a JSON file 'class_weights.json' in
    `data_dir`.

    Weight values are inversely proportional to class frequency, scaled so that
    the largest value is 1.

    To avoid memory issues with computing weights for mask directories with
    many files, `compute_weights` loads masks one at a time and adds their
    class counts to a running talley.

    Args:
        data_dir (str): Name of the directory containing dataset files. Assumes
            the dataset's mask directory is `data_dir`/masks.
        n_classes (int): Number of classes in the masks. This script assumes N
            classes are represented by mask values 0, ..., N-1. In principle,
            this value could be inferred by a pass through the mask arrays,
            but it's easier to just supply it here.


    """
    class_counts = [0] * n_classes
    # Iterate through the masks, adding each mask's class count to the
    # running total
    mask_dir = os.path.join(data_dir, 'label')
    vid_dirs = [d for d in os.listdir(mask_dir)]
    mask_files = [
        os.path.join(mask_dir, vdir, f) 
        for vdir in vid_dirs
        for f in os.listdir(os.path.join(mask_dir, vdir))
        if os.path.splitext(f)[1] == '.png']

    for mask_file in mask_files:
        mask = plt.imread(mask_file)[..., 0] > 0.9  # TODO: Make less hacky
        for i in range(n_classes):
            class_counts[i] += (mask == i).sum()

    # Create weights inversely proportional to frequency
    inv_freq_weights = [sum(class_counts) / count for count in class_counts]
    # Rescale so that the max weight is 1
    ifw_max = max(inv_freq_weights)
    class_weights = [w / ifw_max for w in inv_freq_weights]

    # Save class weights to a JSON
    save_file = os.path.join(data_dir, 'class_weights.json')
    with open(save_file, 'w') as fd:
        json.dump(class_weights, fd)

    print(f'Saved weights\n  {class_weights}\nto {save_file}.')
    return
