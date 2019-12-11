import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from layers import *

class Resnet56(nn.Module):

    def __init__(self, num_classes=10):
        super(Resnet56, self).__init__()

        self.relu = nn.ReLU()

        self.conv1   = MaskedConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(16)

        # ---
        self.conv2  = MaskedConv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2    = nn.BatchNorm2d(16)

        self.conv3  = MaskedConv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3    = nn.BatchNorm2d(16)

        self.conv4  = MaskedConv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4    = nn.BatchNorm2d(16)

        self.conv5  = MaskedConv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5    = nn.BatchNorm2d(16)

        self.conv6  = MaskedConv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6    = nn.BatchNorm2d(16)

        self.conv7  = MaskedConv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7    = nn.BatchNorm2d(16)

        self.conv8  = MaskedConv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8    = nn.BatchNorm2d(16)

        self.conv9  = MaskedConv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9    = nn.BatchNorm2d(16)

        self.conv10 = MaskedConv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10   = nn.BatchNorm2d(16)

        # ---
        self.conv11 = MaskedConv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn11   = nn.BatchNorm2d(32)

        self.conv12 = MaskedConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12   = nn.BatchNorm2d(32)
        
        self.conv13 = MaskedConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn13   = nn.BatchNorm2d(32)
        
        self.conv14 = MaskedConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn14   = nn.BatchNorm2d(32)
        
        self.conv15 = MaskedConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn15   = nn.BatchNorm2d(32)
        
        self.conv16 = MaskedConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn16   = nn.BatchNorm2d(32)
        
        self.conv17 = MaskedConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn17   = nn.BatchNorm2d(32)
        
        self.conv18 = MaskedConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn18   = nn.BatchNorm2d(32)
        
        self.conv19 = MaskedConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn19   = nn.BatchNorm2d(32)
        
        # ---
        self.conv20 = MaskedConv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn20   = nn.BatchNorm2d(64)

        self.conv21 = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn21   = nn.BatchNorm2d(64)

        self.conv22 = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn22   = nn.BatchNorm2d(64)

        self.conv23 = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn23   = nn.BatchNorm2d(64)

        self.conv24 = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn24   = nn.BatchNorm2d(64)

        self.conv25 = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn25   = nn.BatchNorm2d(64)

        self.conv26 = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn26   = nn.BatchNorm2d(64)

        self.conv27 = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn27   = nn.BatchNorm2d(64)

        self.conv28 = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn28   = nn.BatchNorm2d(64)

        # ---
        self.linear1 = MaskedLinear(64, num_classes)




    def setup_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future

        self.conv2.set_mask(torch.Tensor(masks['conv2.weight']))
        self.conv3.set_mask(torch.Tensor(masks['conv3.weight']))
        self.conv4.set_mask(torch.Tensor(masks['conv4.weight']))
        self.conv5.set_mask(torch.Tensor(masks['conv5.weight']))
        self.conv6.set_mask(torch.Tensor(masks['conv6.weight']))
        self.conv7.set_mask(torch.Tensor(masks['conv7.weight']))
        self.conv8.set_mask(torch.Tensor(masks['conv8.weight']))
        self.conv9.set_mask(torch.Tensor(masks['conv9.weight']))

        self.conv10.set_mask(torch.Tensor(masks['conv10.weight']))
        self.conv11.set_mask(torch.Tensor(masks['conv11.weight']))
        self.conv12.set_mask(torch.Tensor(masks['conv12.weight']))
        self.conv13.set_mask(torch.Tensor(masks['conv13.weight']))
        self.conv14.set_mask(torch.Tensor(masks['conv14.weight']))
        self.conv15.set_mask(torch.Tensor(masks['conv15.weight']))
        self.conv16.set_mask(torch.Tensor(masks['conv16.weight']))
        self.conv17.set_mask(torch.Tensor(masks['conv17.weight']))
        self.conv18.set_mask(torch.Tensor(masks['conv18.weight']))
        self.conv19.set_mask(torch.Tensor(masks['conv19.weight']))

        self.conv20.set_mask(torch.Tensor(masks['conv20.weight']))
        self.conv21.set_mask(torch.Tensor(masks['conv21.weight']))
        self.conv22.set_mask(torch.Tensor(masks['conv22.weight']))
        self.conv23.set_mask(torch.Tensor(masks['conv23.weight']))
        self.conv24.set_mask(torch.Tensor(masks['conv24.weight']))
        self.conv25.set_mask(torch.Tensor(masks['conv25.weight']))
        self.conv26.set_mask(torch.Tensor(masks['conv26.weight']))
        self.conv27.set_mask(torch.Tensor(masks['conv27.weight']))
        self.conv28.set_mask(torch.Tensor(masks['conv28.weight']))

        self.linear1.set_mask(torch.Tensor(masks['linear1.weight']))

    def forward(self, x, labels=False, 
                conv1=False, conv2=False, conv3=False, conv4=False, conv5=False, 
                conv6=False, conv7=False, conv8=False, conv9=False, conv10=False, 
                conv11=False, conv12=False, conv13=False, conv14=False, conv15=False,
                conv16=False, conv17=False, conv18=False, conv19=False, conv20=False,
                conv21=False, conv22=False, conv23=False, conv24=False, conv25=False,
                conv26=False, conv27=False, conv28=False, linear1=False):
        # ----
        out = self.relu(self.bn1(self.conv1(x)))
        if conv1:
            return out

        # ----
        outer = self.relu(self.bn2(self.conv2(out)))
        if conv2:
            return outer

        out = self.relu(self.bn3(self.conv3(outer)) + out) 
        if conv3:
            return out
        

        outer = self.relu(self.bn4(self.conv4(out)))
        if conv4:
            return outer

        out = self.relu(self.bn5(self.conv5(outer)) + out) 
        if conv5:
            return out

        
        outer = self.relu(self.bn6(self.conv6(out)))
        if conv6:
            return outer

        out = self.relu(self.bn7(self.conv7(outer)) + out) 
        if conv7:
            return out

        outer = self.relu(self.bn8(self.conv8(out)))
        if conv8:
            return outer

        out = self.relu(self.bn9(self.conv9(outer)) + out) 
        if conv9:
            return out

        outer = self.relu(self.bn10(self.conv10(out)))
        if conv10:
            return outer

        out = self.relu(self.bn11(self.conv11(outer)) + out) 
        if conv11:
            return out

        outer = self.relu(self.bn12(self.conv12(out)))
        if conv12:
            return outer

        out = self.relu(self.bn13(self.conv13(outer)) + out) 
        if conv13:
            return out

        outer = self.relu(self.bn14(self.conv14(out)))
        if conv14:
            return outer

        out = self.relu(self.bn15(self.conv15(outer)) + out) 
        if conv15:
            return out
        outer = self.relu(self.bn16(self.conv16(out)))
        if conv16:
            return outer

        out = self.relu(self.bn17(self.conv17(outer)) + out) 
        if conv17:
            return out
        outer = self.relu(self.bn18(self.conv18(out)))
        if conv18:
            return outer

        out = self.relu(self.bn19(self.conv19(outer)) + out) 
        if conv19:
            return out



        # ----
        #out = self.avgpool(out)
        out = out.view(-1, 512)
        out = self.relu(self.bn14(self.linear1(out)))
        if linear1:
            return out

        out = self.linear3(out)
        #if linear2:
        #    return out

        #out = self.linear3(out)

        if labels:
            out = F.softmax(out, dim=1)

        return out

