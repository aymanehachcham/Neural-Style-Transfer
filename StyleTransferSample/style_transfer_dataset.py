
import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np

class StyleTransferSample(Dataset):

    def __init__(self, image_path=None, tensor_image: torch.Tensor=None, device=None):
        if image_path is not None:
            if os.path.exists(image_path):
                self.image_file = image_path
                self.image = Image.open(self.image_file)
            else:
                raise ValueError('Please give a correct image path')

        if device is None:
            self.device = 'cpu'
            self.image_size = 128

        if device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda:0'
            self.image_size = 512


        # Apply transformations:
        self.tensor_loading = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.tensor_unloading = transforms.ToPILImage()

        if tensor_image is not None:
            tensor_image = tensor_image.cpu().clone()
            tensor_image = tensor_image.squeeze(0)
            self.image = self.tensor_unloading(tensor_image)

        # Processed result to carry on:
        self.processed_image = self.tensor_loading(self.image)
        self.processed_image = self.processed_image.unsqueeze(0).to(self.device)


    def save_image(self):
        self.image.save('Results/output_result.jpg')

    def print_image(self, title=None):
        image = self.image
        if title is not None:
            plt.title = title
        plt.imshow(image)
        plt.pause(5)
        plt.figure()

    def print_processed(self, title='After processing'):
        image = self.processed_image.squeeze(0).detach().cpu()
        image = self.tensor_unloading(image)
        plt.title = title
        plt.imshow(image)
        plt.pause(5)
        plt.figure()



