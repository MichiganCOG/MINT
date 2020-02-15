import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from .layers import *

class VGG16_bn(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG16_bn, self).__init__()

        self.relu = nn.ReLU()

        self.conv1   = MaskedConv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1     = nn.BatchNorm2d(64)
        #self.relu1   = nn.ReLU(inplace=True)

        self.conv2   = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2     = nn.BatchNorm2d(64)
        #self.relu2   = nn.ReLU(inplace=True)

        self.pool1   = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---
        self.conv3   = MaskedConv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        #self.relu3   = nn.ReLU(inplace=True)

        self.conv4   = MaskedConv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4     = nn.BatchNorm2d(128)
        #self.relu4   = nn.ReLU(inplace=True)

        self.pool2   = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ---
        self.conv5   = MaskedConv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5     = nn.BatchNorm2d(256)
        #self.relu5   = nn.ReLU(inplace=True)

        self.conv6   = MaskedConv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6     = nn.BatchNorm2d(256)
        #self.relu6   = nn.ReLU(inplace=True)

        self.conv7   = MaskedConv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn7     = nn.BatchNorm2d(256)
        #self.relu7   = nn.ReLU(inplace=True)

        self.pool3   = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---
        self.conv8   = MaskedConv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn8     = nn.BatchNorm2d(512)
        #self.relu8   = nn.ReLU(inplace=True)

        self.conv9   = MaskedConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn9     = nn.BatchNorm2d(512)
        #self.relu9   = nn.ReLU(inplace=True)

        self.conv10  = MaskedConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn10    = nn.BatchNorm2d(512)
        #self.relu10  = nn.ReLU(inplace=True)

        self.pool4   = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---
        self.conv11  = MaskedConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn11    = nn.BatchNorm2d(512)
        #self.relu11  = nn.ReLU(inplace=True)

        self.conv12  = MaskedConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn12    = nn.BatchNorm2d(512)
        #self.relu12  = nn.ReLU(inplace=True)

        self.conv13  = MaskedConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn13    = nn.BatchNorm2d(512)
        #self.relu13  = nn.ReLU(inplace=True)

        self.pool5   = nn.MaxPool2d(kernel_size=2, stride=2)

        #self.avgpool = nn.AvgPool2d((7, 7))

        self.linear1 = MaskedLinear(512, 512)
        #    nn.ReLU(True),

        self.bn14    = nn.BatchNorm1d(512)
        #self.drop1   = nn.Dropout()
        #self.linear2 = MaskedLinear(4096, 4096)
        #    nn.ReLU(True),

        #self.drop2   = nn.Dropout()
        self.linear3 = MaskedLinear(512, num_classes)


    def setup_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future

        self.conv2.set_mask(torch.Tensor(masks['conv2.weight']).cuda())
        #self.conv3.set_mask(torch.Tensor(masks['conv3.weight']).cuda())
        #self.conv4.set_mask(torch.Tensor(masks['conv4.weight']).cuda())
        #self.conv5.set_mask(torch.Tensor(masks['conv5.weight']).cuda())
        self.conv6.set_mask(torch.Tensor(masks['conv6.weight']).cuda())
        #self.conv7.set_mask(torch.Tensor(masks['conv7.weight']).cuda())
        #self.conv8.set_mask(torch.Tensor(masks['conv8.weight']).cuda())
        self.conv9.set_mask(torch.Tensor(masks['conv9.weight']).cuda())
        self.conv10.set_mask(torch.Tensor(masks['conv10.weight']).cuda())

        self.conv11.set_mask(torch.Tensor(masks['conv11.weight']).cuda())
        self.conv12.set_mask(torch.Tensor(masks['conv12.weight']).cuda())
        self.conv13.set_mask(torch.Tensor(masks['conv13.weight']).cuda())

        self.linear1.set_mask(torch.Tensor(masks['linear1.weight']).cuda())

    def forward(self, x, labels=False, conv1=False, conv2=False, conv3=False, conv4=False, conv5=False, conv6=False, conv7=False, conv8=False, conv9=False, conv10=False, conv11=False, conv12=False, conv13=False, linear1=False, linear2=False, linear3=False):
        # ----
        out = self.relu(self.bn1(self.conv1(x)))
        if conv1:
            return out

        out = self.pool1(self.relu(self.bn2(self.conv2(out))))
        if conv2:
            return out

        # ----
        out = self.relu(self.bn3(self.conv3(out)))
        if conv3:
            return out

        out = self.pool2(self.relu(self.bn4(self.conv4(out))))
        if conv4:
            return out

        # ----
        out = self.relu(self.bn5(self.conv5(out)))
        if conv5:
            return out

        out = self.relu(self.bn6(self.conv6(out)))
        if conv6:
            return out

        out = self.pool3(self.relu(self.bn7(self.conv7(out))))
        if conv7:
            return out

        # ----
        out = self.relu(self.bn8(self.conv8(out)))
        if conv8:
            return out

        out = self.relu(self.bn9(self.conv9(out)))
        if conv9:
            return out

        out = self.pool4(self.relu(self.bn10(self.conv10(out))))
        if conv10:
            return out

        # ----
        out = self.relu(self.bn11(self.conv11(out)))
        if conv11:
            return out

        out = self.relu(self.bn12(self.conv12(out)))
        if conv12:
            return out

        out = self.pool5(self.relu(self.bn13(self.conv13(out))))
        if conv13:
            return out

        # ----
        out = out.view(-1, 512)
        out = self.relu(self.bn14(self.linear1(out)))
        if linear1:
            return out

        out = self.linear3(out)

        if labels:
            out = F.softmax(out, dim=1)

        return out

