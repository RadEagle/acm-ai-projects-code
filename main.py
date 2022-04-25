import os

import constants
from data.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train import starting_train

import torch


def main():
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE}

    # TODO: Add GPU support. This line of code might be helpful.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    # Initalize dataset and model. Then train the model!
    csv_path = 'humpback-whale-identification/train.csv'
    folder_path = 'humpback-whale-identification/train'
    img_size = (128, 128)
    train_dataset = StartingDataset(csv_path, folder_path, img_size)
    val_dataset = StartingDataset(csv_path, folder_path, img_size)
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
