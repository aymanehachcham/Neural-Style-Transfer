
import copy
import torch
import torchvision.models as models
import torch.nn.modules as nn
from Module.loss_modules import ContentLoss, StyleLoss
from Module.normalization import Normalization
from StyleTransferSample.style_transfer_dataset import StyleTransferSample
import torch.optim as optim
import kornia

class StyleTransferModel(nn.Module):
    def __init__(self, content_image, style_image, device=None):
        super(StyleTransferModel, self).__init__()

        if isinstance(content_image, torch.Tensor):
            self.content_image = content_image
        if isinstance(style_image, torch.Tensor):
            self.style_image= style_image

        if device is None:
            self.device = 'cpu'

        if device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda:0'

        self.image_normalization_mean  = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.image_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

        self.vgg19_style_model, _, _ = self.extract_model_and_losses(self.get_layers(self.load_model()))

    def forward(self, content_input):
        output = self.__train_loop(content_input, num_steps=100)
        return output


    def __repr__(self):
        return str(self.vgg19_style_model)

    def load_model(self):
        return models.vgg19(pretrained=True).features.to(self.device).eval()


    def extract_model_and_losses(self, cnn_model):
        global sequential_model, last_layer

        vgg_model = copy.deepcopy(cnn_model)

        content_layers = ['relu3_2']
        style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

        content_losses, style_losses, last_layer = [], [], 0

        style_transfer_model = nn.Sequential(Normalization())

        for count, (name, layer) in enumerate(vgg_model.named_children()):
            style_transfer_model.add_module(name, layer)

            if name in content_layers:
                content_loss = ContentLoss(style_transfer_model(self.content_image))
                style_transfer_model.add_module(f'{name}_ContentLoss', content_loss)
                content_losses.append(content_loss)
                last_layer = count

            if name in style_layers:
                style_loss = StyleLoss(style_transfer_model(self.style_image))
                style_transfer_model.add_module(f'{name}_StyleLoss', style_loss)
                style_losses.append(style_loss)
                last_layer = count

        # Check if the number of layers matches what we want:
        assert len(content_losses) == len(content_layers), 'Not all content layers found'
        assert len(style_losses) == len(style_layers), 'Not all style layers found'

        last_layer += 1 + len(content_losses) + len(style_losses)
        sequential_model = style_transfer_model[:last_layer + 1].to(self.device)

        return sequential_model, content_losses, style_losses


    def get_layers(self, model):
        """Renames VGG model layers to match those in the paper."""
        block, number = 1, 1
        renamed = nn.Sequential()
        for layer in model.children():

            if isinstance(layer, nn.Conv2d):
                name = f'conv{block}_{number}'

            elif isinstance(layer, nn.ReLU):
                name = f'relu{block}_{number}'
                # The inplace ReLU version doesn't play nicely with NST.
                layer = nn.ReLU(inplace=False)
                number += 1

            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{block}'
                # Average pooling was found to generate images of higher quality than
                # max pooling by Gatys et al.
                layer = nn.AvgPool2d(layer.kernel_size, layer.stride)
                block += 1
                number = 1

            else:
                raise RuntimeError(f'Unrecognized layer "{layer.__class__.__name__}""')

            renamed.add_module(name, layer)

        return renamed.to(self.device)


    def __get_optimizer(self, input_tensor: torch.Tensor):
        optimizer = optim.Adam([input_tensor.requires_grad_()], lr=.05)
        return optimizer


    def __train_loop(self, content_input, num_steps=300, style_weight=1000., content_weight=1., log_steps=50):


        batch, channels, height, width = content_input.data.size()
        input_image = content_input.clone()
        input_image = input_image * .01

        # Get the new extracted layers from the model:
        seq_model = self.get_layers(self.load_model())

        model, content_losses, style_losses = self.extract_model_and_losses(seq_model)

        optimizer = self.__get_optimizer(input_image)

        transform = nn.Sequential(kornia.augmentation.RandomResizedCrop(size=(width, height),
                                                                        scale=(.97, 1.),
                                                                        ratio=(.97, 1.03)),
                                  kornia.augmentation.RandomRotation(degrees=1.))

        for epoch in range(num_steps):

            optimizer.zero_grad()
            input_image.data.clamp_(0, 1)
            model(transform(input_image))

            content_loss, style_loss = 0, 0

            for cl in content_losses:
                content_loss += content_weight * cl.loss
            for sl in style_losses:
                style_loss += style_weight * sl.loss

            loss = content_loss + style_loss
            loss.backward()
            optimizer.step()

            if (epoch > 0 and epoch % log_steps == 0) or (epoch + 1) == num_steps:
                print(f'[{epoch}]: content_loss={content_loss.item()},'
                      f' style_loss={style_loss.item():4f}')

        return input_image.data.clamp_(0, 1)
