""" Simple Code To Load MLP Model """

import torch

import torch.nn              as nn
import torchvision.models    as models
import torch.nn.functional   as F
import torch.utils.model_zoo as model_zoo

from torch.autograd import Variable

# Custom Imports
from .layers import *

class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2_drop = nn.Dropout(0.2)

        self.fc1 = MaskedLinear(28*28, 500, act='relu')
        self.fc2 = MaskedLinear(500, 300,   act='relu')
        self.fc3 = MaskedLinear(300,        num_classes)

    def forward(self, x, labels=False):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.fc1_drop(x)
        x = self.fc2(x)
        x = self.fc2_drop(x)
        x = self.fc3(x)

        if labels:
            x = F.softmax(x, dim=1)

        # END IF

        return x

    def setup_masks(self, mask):
        self.fc2.set_mask(torch.Tensor(mask['fc2.weight']).cuda())
        self.fc3.set_mask(torch.Tensor(mask['fc3.weight']).cuda())
