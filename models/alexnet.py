import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from layers import *
__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class Alexnet(nn.Module):

    #def __init__(self, num_classes=1000):
    #    super(Alexnet, self).__init__()
    #    self.features = nn.Sequential(
    #        nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
    #        nn.ReLU(inplace=True),
    #        nn.MaxPool2d(kernel_size=3, stride=2),
    #        nn.Conv2d(64, 192, kernel_size=5, padding=2),
    #        nn.ReLU(inplace=True),
    #        nn.MaxPool2d(kernel_size=3, stride=2),
    #        nn.Conv2d(192, 384, kernel_size=3, padding=1),
    #        nn.ReLU(inplace=True),
    #        nn.Conv2d(384, 256, kernel_size=3, padding=1),
    #        nn.ReLU(inplace=True),
    #        nn.Conv2d(256, 256, kernel_size=3, padding=1),
    #        nn.ReLU(inplace=True),
    #        nn.MaxPool2d(kernel_size=3, stride=2),
    #    )
    #    self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
    #    self.classifier = nn.Sequential(
    #        nn.Dropout(),
    #        nn.Linear(256 * 6 * 6, 4096),
    #        nn.ReLU(inplace=True),
    #        nn.Dropout(),
    #        nn.Linear(4096, 4096),
    #        nn.ReLU(inplace=True),
    #        nn.Linear(4096, num_classes),
    #    )

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
        #self.linear1 = MaskedLinear(256 * 6 * 6, 4096)
        self.linear1 = MaskedLinear(256 * 6 * 6, 8)
        self.relu6   = nn.ReLU(inplace=True)
        self.drop2   = nn.Dropout()
        #self.linear2 = MaskedLinear(4096, 4096)
        self.linear2 = MaskedLinear(8, 8)
        self.relu7   = nn.ReLU(inplace=True)
        #self.linear3 = MaskedLinear(4096, 10)
        self.linear3 = MaskedLinear(8, 10)

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

    def forward(self, x, labels=False, linear1=False, linear2=False, linear3=False):
        out = self.pool1(self.relu1(self.conv1(x)))
        out = self.pool2(self.relu2(self.conv2(out)))
        out = self.pool3(self.relu5(self.conv5(self.relu4(self.conv4(self.relu3(self.conv3(out)))))))

        out = self.avgpool(out)
 
        out = self.relu6(self.linear1(self.drop1(out.view(-1, 256*6*6))))
        if linear1:
            return out

        out = self.relu7(self.linear2(self.drop2(out)))
        if linear2:
            return out

        out = self.linear3(out)
        if linear3:
            return out

        if labels:
            out = F.softmax(out, dim=1)

        return out
