import torch
import torch.nn as nn
import torchvision.models.vgg as vgg
import torch.utils.model_zoo as model_zoo

__all__ = ['vggd2', 'vgge2']

cfg = {
    'A': [64 , 'R', 'M',
          128, 'R', 'M',
          256, 'R', 256, 'R', 'M',
          512, 'R', 512, 'R', 'M',
          512, 'R', 512, 'R', 'M'],
    'B': [64 , 'R', 64, 'R', 'M',
          128, 'R', 128, 'R', 'M',
          256, 'R', 256, 'R', 'M',
          512, 'R', 512, 'R', 'M',
          512, 'R', 512, 'R', 'M'],
    'D': [64 , 'R', 64, 'R', 'M',
          128, 'R', 128, 'R', 'M',
          256, 'R', 256, 'R', 256, 'R', 'M',
          512, 'R', 512, 'R', 512, 'R', 'M',
          512, 'R', 512, 'R', 512, 'R', 'M'],
    'D2':[64 , 'No', 64 , 'R', 'M',
          128, 'R', 128, 'R', 'M',
          256, 'R', 256, 'R', 256, 'R', 'M',
          512, 'R', 512, 'R', 512, 'R', 'M',
          512, 'R', 512, 'R', 512, 'R', 'M'],
    'E2': [64 ,'No', 64 ,'R', 'M',
          128, 'R', 128, 'R', 'M',
          256, 'R', 256, 'R', 256, 'R', 256, 'R', 'M',
          512, 'R', 512, 'R', 512, 'R', 512, 'R', 'M',
          512, 'R', 512, 'R', 512, 'R', 512, 'R', 'M'],
}


class NoOp(nn.Module):
    def forward(self, x):
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'R':
            layers += [nn.ReLU(inplace=True)]
        elif v == 'No':
            layers += [NoOp()]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v)]
            else:
                layers += [conv2d]
            in_channels = v
    return nn.Sequential(*layers)


def vggd2(pretrained=True):
    model = vgg.VGG(make_layers(cfg['D2']), init_weights=(not pretrained))
    if pretrained:
        model.load_state_dict(model_zoo.load_url(vgg.model_urls['vgg16']))
        # Turn gradients off for all parameters
        for p in model.parameters():
            p.requires_grad = False
        # Turn back on for a couple
        for l in (model.features[0], model.features[2]):
            for p in l.parameters():
                p.requires_grad = True
    return model


def vgge2(pretrained=True):
    model = vgg.VGG(make_layers(cfg['D2'], init_weights=(not pretrained)))
    if pretrained:
        model.load_state_dict(model_zoo.load_url(vgg.model_urls['vgg19']))
        # Turn gradients off for all parameters
        for p in model.parameters():
            p.requires_grad = False
        # Turn back on for a couple
        for l in (model.features[0], model.features[2]):
            for p in l.parameters():
                p.requires_grad = True
    return model

