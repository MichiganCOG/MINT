import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from layers import *

class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc1 = MaskedLinear(28*28, 500)
        self.fc2 = MaskedLinear(500, 300)
        self.fc3 = MaskedLinear(300, num_classes)

    def forward(self, x, labels=False, fc1=False, fc2=False):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        if fc1:
            return x

        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        if fc2:
            return x

        x = self.fc2_drop(x)
        x = self.fc3(x)

        if labels:
            x = F.softmax(x, dim=1)

        return x
