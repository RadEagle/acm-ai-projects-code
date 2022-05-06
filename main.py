import os

import constants
from data.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train import starting_train
import numpy.random as random

import torch
import torch.utils.data as td


def main():
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE}

    # TODO: Add GPU support. This line of code might be helpful.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)
    print("Device used:", device)

    # Initalize dataset and model. Then train the model!
    csv_path = 'humpback-whale-identification/train.csv'
    folder_path = 'humpback-whale-identification/train'
    img_size = (128, 128)
    train_dataset = StartingDataset(csv_path, folder_path, img_size)
    val_size = 0.15
    val_indices = random.choice(range(len(train_dataset)), size=int(len(train_dataset)*val_size), replace=False)
    val_dataset = td.Subset(train_dataset, val_indices)
    model = StartingNetwork()
    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=constants.N_EVAL,
        device=device
    )


if __name__ == "__main__":
    main()
