
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from Module.style_transfer_model import StyleTransferModel
from StyleTransferSample.style_transfer_dataset import StyleTransferSample


style_path = 'Data/Style_samples/cubic-picasso2.jpg'
content_path = 'Data/Content_samples/aymane.png'

content = StyleTransferSample(content_path, device='cuda')
style = StyleTransferSample(style_path, device='cuda')

content_input = content.processed_image
style_input = style.processed_image

transfer_model = StyleTransferModel(content_input, style_input, device='cuda')

output = transfer_model(content_input)

tensor_unloading = transforms.ToPILImage()
image = tensor_unloading(output.squeeze(0).detach().cpu())
image.save('Results/output_result.jpg')