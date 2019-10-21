'''ResNet18/34/50/101/152 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, labels=False, features=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        if features:
            return out/torch.norm(out,2,1).unsqueeze(1)

        out = self.linear(out)

        if labels:
            out = F.softmax(out, dim=1)

        return out

class ResNet18_Expanded(nn.Module):

    def __init__(self, num_classes):
        super(ResNet18_Expanded, self).__init__()

        self.conv1  = MaskedConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)

        # ----
        self.conv2  = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(64)

        self.conv3  = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3    = nn.BatchNorm2d(64)

        self.conv4  = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4    = nn.BatchNorm2d(64)

        self.conv5  = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5    = nn.BatchNorm2d(64)
        
        # ----
        self.conv6  = MaskedConv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn6    = nn.BatchNorm2d(128)

        self.conv7  = MaskedConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7    = nn.BatchNorm2d(128)

        self.conv8  = MaskedConv2d(64, 128, kernel_size=1, stride=2, bias=False)
        self.bn8    = nn.BatchNorm2d(128)

        self.conv9  = MaskedConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9    = nn.BatchNorm2d(128)
        
        self.conv10 = MaskedConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10   = nn.BatchNorm2d(128)

        # ----
        self.conv11 = MaskedConv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn11   = nn.BatchNorm2d(256)

        self.conv12 = MaskedConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12   = nn.BatchNorm2d(256)

        self.conv13 = MaskedConv2d(128, 256, kernel_size=1, stride=2, bias=False)
        self.bn13   = nn.BatchNorm2d(256)

        self.conv14 = MaskedConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn14   = nn.BatchNorm2d(256)
        
        self.conv15 = MaskedConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn15   = nn.BatchNorm2d(256)

        # ----
        self.conv16 = MaskedConv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn16   = nn.BatchNorm2d(512)

        self.conv17 = MaskedConv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn17   = nn.BatchNorm2d(512)

        self.conv18 = MaskedConv2d(256, 512, kernel_size=1, stride=2, bias=False)
        self.bn18   = nn.BatchNorm2d(512)

        self.conv19 = MaskedConv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn19   = nn.BatchNorm2d(512)
        
        self.conv20 = MaskedConv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn20   = nn.BatchNorm2d(512)

        self.linear = MaskedLinear(512, num_classes)

    def forward(self, x, labels=False):
        out = F.relu(self.bn1(self.conv1(x)))

        # ----
        out = F.relu(self.bn2(self.conv2(out)))

        out = F.relu(self.bn3(self.conv3(out)))

        out = F.relu(self.bn4(self.conv4(out)))

        out = F.relu(self.bn5(self.conv5(out)))

        # ----
        temp_x = out

        out = F.relu(self.bn6(self.conv6(out)))

        out = self.bn7(self.conv7(out))

        out = F.relu(self.bn8(self.conv8(temp_x)) + out)

        out = F.relu(self.bn9(self.conv9(out)))

        out = F.relu(self.bn10(self.conv10(out)))

        # ----
        temp_x = out

        out = F.relu(self.bn11(self.conv11(out)))

        out = self.bn12(self.conv12(out))

        out = F.relu(self.bn13(self.conv13(temp_x)) + out)

        out = F.relu(self.bn14(self.conv14(out)))

        out = F.relu(self.bn15(self.conv15(out)))

        # ----
        temp_x = out

        out = F.relu(self.bn16(self.conv16(out)))

        out = self.bn17(self.conv17(out))

        out = F.relu(self.bn18(self.conv18(temp_x)) + out)

        out = F.relu(self.bn19(self.conv19(out)))

        out = F.relu(self.bn20(self.conv20(out)))

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        out = self.linear(out)

        if labels:
            out = F.softmax(out, dim=1)

        return out

def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)

def resnet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes)

def resnet50(num_classes=10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes)

def resnet101(num_classes=10):
    return ResNet(Bottleneck, [3,4,23,3], num_classes)

def resnet152(num_classes=10):
    return ResNet(Bottleneck, [3,8,36,3], num_classes)

def test_resnet():
    net = resnet50()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test_resnet()
