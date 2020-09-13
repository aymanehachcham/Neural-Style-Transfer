from normalization import Normalization
from loss_modules import Contentloss, Styleloss
from custom_dataset import StyleTransferImage
import torchvision.models as models
import torch.nn as nn
import copy
import torch


class VGGStyleTransfer(nn.Module):
    """

    """
    def __init__(self, root_dir, content_image_path, style_image_path, device=None):
        super(VGGStyleTransfer, self).__init__()
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.mean = vgg_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.std = vgg_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        self.vgg_model = models.vgg19(pretrained=True).features.to(device).eval()
        self.content_image = StyleTransferImage(content_image_path, root_dir, self.device)
        self.style_image = StyleTransferImage(style_image_path, root_dir, self.device)
        self.input_img = self.content_image.image_tensor.clone()


    def extract_layers(self, model, content_layers: list, style_layers: list):
        model_copy = copy.deepcopy(model)
        content_modules = []
        style_modules = []

        for number, layer in model_copy.named_children():
            if isinstance(layer, nn.Conv2d):
                name = 'conv_{}'.format(number)
                if name in content_layers:
                    content_modules.append(layer)
                elif name in style_layers:
                    style_modules.append(layer)
            else:
                continue

        return content_modules, style_modules

    def assemble_model(self, model, content_image: StyleTransferImage, style_image: StyleTransferImage):
        model_copy = copy.deepcopy(model)

        content_layers = ['conv_4']
        style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        content_loss = []
        style_loss = []

        normalization = Normalization(self.mean, self.std).to(self.device)
        vgg_model = nn.Sequential(normalization)

        i = 0
        for layer in model_copy.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
                vgg_model.add_module(name, layer)

                if name in content_layers:
                    target = vgg_model(content_image.image_tensor).detach()
                    cont_loss = Contentloss(target)
                    vgg_model.add_module('content_loss_{}'.format(i), cont_loss)
                    content_loss.append(cont_loss)

                if name in style_layers:
                    target_f = vgg_model(style_image.image_tensor).detach()
                    sty_loss = Styleloss(target_f)
                    vgg_model.add_module('style_loss_{}'.format(i), sty_loss)
                    style_loss.append(sty_loss)

            elif isinstance(layer, nn.ReLU):
                name = 'Relu_{}'.format(i)
                layer = nn.Relu(inplace=False)


            elif isinstance(layer, nn.MaxPool2d):
                name = 'AvgPool2d_{}'.format(i)
                layer = nn.AvgPool2d(kernel_size=2, stride=2)

            elif isinstance(layer, nn.BatchNorm2d):
                name = 'BatchNorm_{}'.format(i)
            else:
                raise RuntimeError('Module not found : {}'.format(layer.__class__.__name__))

            vgg_model.add_module(layer, name)

            if name == 'Relu_5':
                break

        return vgg_model, style_loss, content_loss

    @staticmethod
    def closure(model: torch.nn.Module, input_image: torch.Tensor, optimizer: torch.optim, style_loss: list, content_loss: list):
        # Correct the input image: values between 0 and 1
        input_image = torch.clamp(input_image, min=0, max=1)
        optimizer.zero_grad()
        style_weight = 1000000
        content_weight = 1
        style_score = content_score = 0

        # Computing the Style and Content Losses:
        output = model(input_image)
        for sty_loss in style_loss:
            style_score += sty_loss
        for cont_loss in content_loss:
            content_score += cont_loss

        style_score *= style_weight
        content_score *= content_weight
        loss = style_score + content_score
        loss.backward()

        return loss

    def train_loop(self, epochs=300):
        i = 0
        optimizer = torch.optim.LBFGS([self.input_img.requires_grad_()])
        model, style_loss, content_loss = self.assemble_model(self.vgg_model, self.content_image, self.style_image)
        while i <= epochs:
            optimizer.step(VGGStyleTransfer.closure(model, self.input_img, optimizer, style_loss, content_loss))
            i += 1

        self.input_img.data.clamp_(0, 1)
        return self.input_image

    def forward(self):
        # Building the model:
        output = self.train_loop()
        return output


