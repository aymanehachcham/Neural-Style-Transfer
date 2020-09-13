import torch
import torch.nn as nn


class Normalization(nn.Module):
    """
    Normalize image tensor before fed to VGG neural network
    """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)


    def forward(self, input_tensor):
        return (input_tensor - self.mean) / self.std

    