"""Stripped-down minimal script to load in the challenge data and train a
U-Net on it. For dataset exploration and testing, data exclusion decision
rationale, and visual progress updates during the training loop, launch the
Jupyter notebook main.ipynb. Coding challenge questions are also answered in
the 'Q&A' section at the top of main.ipynb.

"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import jaccard_score
from torch import nn
from torch.utils.data import DataLoader, random_split

# Local project files
from src.ChallengeDataset import ChallengeDataset
from src.UNet import UNet
from src.generate_weights import generate_weights


def main(data_dir: str,
         n_epochs: int,
         device: str) -> nn.Module:
    """

    Args:
        data_dir (str): Directory containing a dataset's image, mask, and
            weight data. Assumes a directory structure of:
                data_dir/
                    images/
                    masks/
        n_epochs (int): Number of epochs to train for
        device (str): PyTorch device to train on.

    Returns:
        (nn.Module): A trained U-Net

    """
    print('''
This is a minimal script to load in the challenge data and train a 
U-Net on it. For dataset exploration and testing, data exclusion decision 
rationale, and visual progress updates during the training loop, launch the 
Jupyter notebook main.ipynb. Coding challenge questions are also answered in 
the 'Q&A' section at the top of main.ipynb.
    
    ''')

    # Generate weights if necessary

    class_weight_file = 'class_weights.json'
    if class_weight_file not in os.listdir(data_dir):
        generate_weights(data_dir, n_classes=3)

    # Load data

    dataset = ChallengeDataset(data_dir)
    print(f'Loaded {len(dataset)} image+mask pairs')

    # Create a random train+val split
    n_data = len(dataset)
    n_train = int(.8 * n_data)
    n_val = n_data - n_train
    train_dataset, val_dataset = random_split(dataset, (n_train, n_val))
    print(f'Created {len(train_dataset)}-{len(val_dataset)} train-val split.\n')

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=1)
    val_loader = DataLoader(
        val_dataset,
        shuffle=True,
        batch_size=1)

    # Build model

    model = UNet(
        in_channels=1,
        n_classes=3,
        depth=4,
        n_init_filters=32,
        padding=True,
        instance_norm=True,
        up_mode='upconv',
        leaky=True)
    model.to(device)

    print(f'Loaded U-Net to {device}\n')

    # Build optimizer

    optim = torch.optim.Adam(
        model.parameters(),
        lr=10 ** -3.3,
        weight_decay=1e-5)

    # Define loss function

    def loss_function(prediction, label, weight):
        xentropy = F.cross_entropy(
            prediction, label, reduction='none')
        return torch.sum(torch.mul(weight, xentropy))

    # Validation reporting settings
    epochs_per_eval = 25

    # Main training loop

    for epoch in range(n_epochs):
        time0 = time.time()

        # Train

        model.train()

        losses = []

        for i, (image, label, weight) in enumerate(train_loader):
            image = image.to(device).unsqueeze(1)
            label = label.long().to(device)
            weight = weight.to(device)

            prediction = model(image)

            loss = loss_function(prediction, label, weight)
            losses.append(loss.item())

            optim.zero_grad()
            loss.backward()
            optim.step()

        train_loss = sum(losses) / len(losses)

        time1 = time.time()

        # Validate

        if epoch % epochs_per_eval == 0 or epoch == n_epochs - 1:
            model.eval()
            val_losses = []
            predictions = []
            labels = []

            # Produce validation predictions, compute loss and MIoU

            for image, label, weight in val_loader:
                image = image.to(device).unsqueeze(1)
                label = label.long().to(device)
                weight = weight.to(device)

                prediction = model(image)

                classes = torch.argmax(prediction, dim=1).cpu()
                predictions.append(np.squeeze(classes).flatten())
                labels.append(label.cpu().numpy().flatten())

                val_losses.append(loss_function(prediction, label, weight))

            val_loss = sum(val_losses) / len(val_losses)

            miou = jaccard_score(
                np.concatenate(labels).flatten(),
                np.concatenate(predictions).flatten(),
                average='macro')

            time2 = time.time()

            print(f'Epoch {epoch + 1} validation:\n'
                  f'  Train time: {time1 - time0:.2f}s train  '
                  f'  Val time:  {time2 - time1:.2f}s\n'
                  f'  Train loss: {train_loss:.4f}  '
                  f'  Val loss: {val_loss:.4f}  '
                  f'  Val MIoU: {miou:.3f}\n')

    return model


if __name__ == '__main__':
    data_dir = os.path.join('.', 'data', 'cleaned')
    n_epochs = 1
    device = 'cuda'

    args = sys.argv[1:]
    if len(args) > 0:
        data_dir = args[0]
    if len(args) > 1:
        n_epochs = int(args[1])
    if len(args) > 2:
        device = args[2]

    trained_model = main(data_dir, n_epochs, device)
