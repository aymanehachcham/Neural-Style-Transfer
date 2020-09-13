import torch
import torch.nn as nn
import torch.nn.functional as F

class Contentloss(nn.Module):
    """
    The Content Loss module responsible for calculating over each iteration
    the loss between the original content image and the current state
    of the convolutional layers involved
    Uses: Mean Squared Error: Features(Original Content Image), Features(Conv Layers for Content)
    Added as a transparent layer at the end of convolution layer
    """
    def __init__(self, target):
        super(Contentloss, self).__init__()

        self.target = target.detach()

    def forward(self, input_fm_layer):
        self.content_loss = F.mse_loss(input_fm_layer, self.target)
        return input_fm_layer


class Styleloss(nn.Module):
    """
    The Style Loss module act similarly to the Content Loss. A transparent layer inserted
    after each Convolutional layer associated with style extraction.
    Compute the Gram Matrix of the layer's feature maps and calculate the MSE loss with the target
    """
    def __init__(self, target):
        super(Styleloss, self).__init__()

        self.target = self.gram_matrix(target).detach()

    def forward(self, input_fm_layer):
        gram_fm_layer = self.gram_matrix(input_fm_layer)
        self.style_loss = F.mse_loss(gram_fm_layer, self.target)
        return input_fm_layer

    def gram_matrix(self, input):
        """
        Define the Gram matrix for an input
        :param input: The tensor holding the feature maps of the layer
        :return: the gram matrix of the latter feature maps
        """
        # Obtain the dimensions of the layer's feature maps
        batch, channels, height, width = input.size()
        # Define the new dimensions: F(xl) => \hatF(xl)
        new_height = batch*channels
        new_width = height*width

        # Reshape the tensor with the new dimensions and calculate the gram product
        input_new_dim = input.view(new_height, new_width)
        gram_matrix = torch.mm(input_new_dim, input_new_dim.t())

        # Normalize the gram matrix
        return gram_matrix.div(new_width*new_width)






