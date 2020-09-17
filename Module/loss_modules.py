import torch
import torch.nn as nn
from torch.nn import functional as F


# Loss Modules
class ContentLoss(nn.Module):

    def __init__(self, target,):
        super().__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self._target = target.detach()
        self.loss = None

    def forward(self, input):
        self.loss = F.l1_loss(input, self._target)
        return input



class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super().__init__()
        self._target = self.__gram_matrix(target_feature.detach()).detach()
        self.loss = None

    def __gram_matrix(self, input):
        """Returns the normalized Gram matrix of the input."""
        n, c, w, h = input.size()
        features = input.view(n * c, w * h)
        G = torch.mm(features, features.t())
        return G.div(n * c * w * h)

    def forward(self, input):
        G = self.__gram_matrix(input)
        self.loss = F.l1_loss(G, self._target)
        return input









