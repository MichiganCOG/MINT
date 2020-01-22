import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

from torch.utils.model_zoo import load_url as load_state_dict_from_url
from .layers import *
__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class Alexnet(nn.Module):

    def __init__(self, num_classes=1000):
        super(Alexnet, self).__init__()

        self.conv1   = MaskedConv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.relu1   = nn.ReLU(inplace=True)
        self.pool1   = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2   = MaskedConv2d(64, 192, kernel_size=5, padding=2)
        self.relu2   = nn.ReLU(inplace=True)
        self.pool2   = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3   = MaskedConv2d(192, 384, kernel_size=3, padding=1)
        self.relu3   = nn.ReLU(inplace=True)
        self.conv4   = MaskedConv2d(384, 256, kernel_size=3, padding=1)
        self.relu4   = nn.ReLU(inplace=True)
        self.conv5   = MaskedConv2d(256, 256, kernel_size=3, padding=1)
        self.relu5   = nn.ReLU(inplace=True)
        self.pool3   = nn.MaxPool2d(kernel_size=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.drop1   = nn.Dropout()
        self.linear1 = MaskedLinear(256 * 6 * 6, 4096)
        self.relu6   = nn.ReLU(inplace=True)
        self.drop2   = nn.Dropout()
        self.linear2 = MaskedLinear(4096, 4096)
        self.relu7   = nn.ReLU(inplace=True)
        self.linear3 = MaskedLinear(4096, 1000)

    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future

        self.conv1.set_mask(masks[0])
        self.conv2.set_mask(masks[1])
        self.conv3.set_mask(masks[2])
        self.conv4.set_mask(masks[3])
        self.conv5.set_mask(masks[4])

        self.linear1.set_mask(masks[5])
        self.linear2.set_mask(masks[6])
        self.linear3.set_mask(masks[7])

    def forward(self, x, labels=False):
        out = self.pool1(self.relu1(self.conv1(x)))

        out = self.pool2(self.relu2(self.conv2(out)))

        out = self.relu3(self.conv3(out))

        out = self.relu4(self.conv4(out))

        out = self.pool3(self.relu5(self.conv5(out)))

        out = self.avgpool(out)
 
        out = self.relu6(self.linear1(self.drop1(out.view(-1, 256*6*6))))

        out = self.relu7(self.linear2(self.drop2(out)))

        out = self.linear3(out)

        if labels:
            out = F.softmax(out, dim=1)

        return out

def alexnet(num_classes):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = Alexnet(num_classes=num_classes)
    state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                          progress=True)

    new_state_dict = {}
    new_state_dict['conv1.weight'] = state_dict['features.0.weight']
    new_state_dict['conv1.bias']   = state_dict['features.0.bias']
    new_state_dict['conv2.weight'] = state_dict['features.3.weight']
    new_state_dict['conv2.bias']   = state_dict['features.3.bias']
    new_state_dict['conv3.weight'] = state_dict['features.6.weight']
    new_state_dict['conv3.bias']   = state_dict['features.6.bias']
    new_state_dict['conv4.weight'] = state_dict['features.8.weight']
    new_state_dict['conv4.bias']   = state_dict['features.8.bias']
    new_state_dict['conv5.weight'] = state_dict['features.10.weight']
    new_state_dict['conv5.bias']   = state_dict['features.10.bias']

    new_state_dict['linear1.weight'] = state_dict['classifier.1.weight']
    new_state_dict['linear1.bias']   = state_dict['classifier.1.bias']
    new_state_dict['linear2.weight'] = state_dict['classifier.4.weight']
    new_state_dict['linear2.bias']   = state_dict['classifier.4.bias']
    new_state_dict['linear3.weight'] = state_dict['classifier.6.weight']
    new_state_dict['linear3.bias']   = state_dict['classifier.6.bias']
    del state_dict

    model.load_state_dict(new_state_dict)
    return model
