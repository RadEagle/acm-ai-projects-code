import torch
import torch.nn as nn


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # self.fc = nn.Linear(224 * 224 * 3, 1)
        # self.sigmoid = nn.Sigmoid()

        # also try resnet101, resnet151
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.resnetfc = nn.Linear(2048, 5)

    def forward(self, x):
        # x = self.flatten(x)
        # x = self.fc(x)
        # x = self.sigmoid(x)

        x = self.newmodel(x)
        x = self.flatten(x)
        x = self.resnetfc(x)
        return x
