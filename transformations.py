import torch
from skimage import io, transform

class Rescale(object):
    """

    """
    def __init__(self, default_size=128):
        assert isinstance(default_size, int)
        if torch.cuda.is_available():
            self.image_size = 512
        self.image_size = default_size

    def __call__(self, image):
        new_height, new_width = int(self.image_size), int(self.image_size)

        image = transform.resize(image, (new_height, new_width))
        return image

