import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pytorch_metric_learning import losses, miners

# extra imports
from torch.utils.tensorboard import SummaryWriter


def create_label_dict(labels, label_dict):
    # count = len(book) + 1o
    book = label_dict
    count = len(book)
    for label in labels:
        if label not in book:
            book[label] = count
            count += 1
    return book


def key_to_value(labels, book):
    return tuple(book[label] for label in labels)


def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval, device):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

    # set data points before evaluating
    data_points_before_test = 10

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # create a label string to int dictionary
    label_dict = {}

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters(), lr = 0.05)
    # loss_fn = nn.CrossEntropyLoss()

    margin = 1

    loss_fn = losses.TripletMarginLoss(margin=margin)
    miner = miners.BatchEasyHardMiner(pos_strategy='all', neg_strategy='hard')

    step = 0
    writer = SummaryWriter()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        print("working")

        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):
            # TODO: Backpropagation and gradient descent
            model.train()
            images, labels = batch
            images = images.to(device)
            
            # TODO: convert labels from string to int
            label_dict = create_label_dict(labels, label_dict)
            labels = key_to_value(labels, label_dict)

            labels = torch.tensor(labels).to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            embeddings = model(images) # images is a batch of images
            print(f"labels: {labels}\nembeddings: {embeddings}")
            hard_triplets = miner(embeddings, labels)
            loss = loss_fn(embeddings, labels, hard_triplets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # total_less += loss.item()
            step += 1

            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                # TODO:
                # Compute training loss and accuracy.
                accuracy = compute_accuracy(embeddings, labels)

                # DEBUG: Log accuracy and loss directly to the terminal
                # print(f"accuracy: {accuracy}")
                # print(f"loss: {loss}")

                # Log the results to Tensorboard.
                writer.add_scalar("Accuracy", accuracy, step)
                writer.add_scalar("Loss", loss, step)

                # Don't forget to turn off gradient calculations!

            if step % (n_eval * data_points_before_test) == 0:
                eval_step = step / (n_eval * data_points_before_test)
                evaluate(val_loader, model, loss_fn, miner, device, writer, label_dict, eval_step)

        print()


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """

    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn, miner, device, writer, label_dict, step):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    TODO!
    """
    # INITIAL CODE, MAY NOT BE FINAL
    print(len(val_loader))
    step_ = 119 * (step - 1)
    with torch.no_grad():
        for batch in tqdm(val_loader, position=0, leave=False):
            step_ += 1
            images, labels = batch
            images = images.to(device)
            label_dict = create_label_dict(labels, label_dict)
            labels = key_to_value(labels, label_dict)
            labels = torch.tensor(labels).to(device)
            outputs = model(images)
            hard_triplets = miner(outputs, labels)
            loss = loss_fn(outputs, labels, hard_triplets)
            outputs = torch.argmax(outputs, dim=1)
            outputs = torch.tensor(outputs, dtype=torch.float)
            labels = torch.tensor(labels, dtype=torch.float)
            accuracy = compute_accuracy(outputs, labels)

            writer.add_scalar("Eval Accuracy", accuracy, step_)
            writer.add_scalar("Eval Loss", loss, step_)
