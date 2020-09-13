from __future__ import print_function

import torch
# from custom_dataset import StyleTransferImage
# import torchvision.models as models
# import torch.nn as nn
# import copy
# from loss_modules import Contentloss, Styleloss
# from vgg_style_transfer import VGGStyleTransfer

device = "cpu"
if torch.cuda.is_available():
    print(torch.cuda.get_device_name())

# root_dir = 'C:\\Users\\ayman\\OneDrive\\Documents\\Neural Style Transfer\\Style Transfer Modules'
# content_img = 'dancing.jpg'
# style_img = 'picasso.jpg'
#
# content = StyleTransferImage(content_img, root_dir, device='cuda')
# style = StyleTransferImage(style_img, root_dir, device='cuda')
#
# #style.print_image()
#
# output = VGGStyleTransfer(root_dir, content_img, style_img, device="cuda")
#
# output.train_loop()