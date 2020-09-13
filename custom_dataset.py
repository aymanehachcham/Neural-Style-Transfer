from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch

from skimage import io
import matplotlib.pyplot as plt
import os

from transformations import Rescale


class StyleTransferImage(Dataset):

    def __init__(self, image_file, root_dir, device=None):
        """

        :param image_file:
        :param root_dir:
        :param device:
        """
        self.image_path = os.path.join(root_dir, image_file)
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        if torch.cuda.is_available():
            image_size = 512
        else:
            image_size = 128

        self.transformer = transforms.Compose([
            Rescale(image_size),
            transforms.ToTensor()
        ])

        self.unload_image = transforms.ToPILImage()

        image = io.imread(self.image_path)
        image = self.transformer(image).unsqueeze(0)
        self.image_tensor = image.to(self.device, torch.float)


    def __call__(self, *args, **kwargs):
        return self.image   _tensor

    def __len__(self):
        return self.image_tensor.squeeze(0).shape

    def to_image(self, tensor: torch.Tensor, title=None):
        image = self.unload_image(tensor)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(10)
        plt.figure()

    def print_image(self, title=None):
        image = self.image_tensor.cpu().clone()
        image = image.squeeze(0)
        image = self.unload_image(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
            plt.pause(10)
            plt.figure()

