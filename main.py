
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from Module.style_transfer_model import StyleTransferModel
from StyleTransferSample.style_transfer_dataset import StyleTransferSample


style_path = 'Data/Style_samples/picasso.jpg'
content_path = 'Data/Content_samples/aya.jpg'

content = StyleTransferSample(content_path, device='cuda')
style = StyleTransferSample(style_path, device='cuda')

content_input = content.processed_image
style_input = style.processed_image

transfer_model = StyleTransferModel(content_input, style_input, device='cuda')

print(transfer_model)
print(transfer_model(content_input))

traced_style_model = torch.jit.trace(transfer_model, content_input)
print(transfer_model)
