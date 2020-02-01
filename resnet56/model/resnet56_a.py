import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from .layers import *

class Resnet56_A(nn.Module):

    def __init__(self, num_classes=10):
        super(Resnet56_A, self).__init__()

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

        self.conv11 = MaskedConv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11   = nn.BatchNorm2d(16)

        self.conv12 = MaskedConv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12   = nn.BatchNorm2d(16)
        
        self.conv13 = MaskedConv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn13   = nn.BatchNorm2d(16)
        
        self.conv14 = MaskedConv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn14   = nn.BatchNorm2d(16)
        
        self.conv15 = MaskedConv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn15   = nn.BatchNorm2d(16)
        
        self.conv16 = MaskedConv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn16   = nn.BatchNorm2d(16)
        
        self.conv17 = MaskedConv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn17   = nn.BatchNorm2d(16)
        
        self.conv18 = MaskedConv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn18   = nn.BatchNorm2d(16)
        
        self.conv19 = MaskedConv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn19   = nn.BatchNorm2d(16)
        
        # ---
        self.conv20 = MaskedConv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn20   = nn.BatchNorm2d(32)

        self.conv21 = MaskedConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn21   = nn.BatchNorm2d(32)

        self.conv22 = MaskedConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn22   = nn.BatchNorm2d(32)

        self.conv23 = MaskedConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn23   = nn.BatchNorm2d(32)

        self.conv24 = MaskedConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn24   = nn.BatchNorm2d(32)

        self.conv25 = MaskedConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn25   = nn.BatchNorm2d(32)

        self.conv26 = MaskedConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn26   = nn.BatchNorm2d(32)

        self.conv27 = MaskedConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn27   = nn.BatchNorm2d(32)

        self.conv28 = MaskedConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn28   = nn.BatchNorm2d(32)

        self.conv29 = MaskedConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn29   = nn.BatchNorm2d(32)

        self.conv30 = MaskedConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn30   = nn.BatchNorm2d(32)

        self.conv31 = MaskedConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn31   = nn.BatchNorm2d(32)

        self.conv32 = MaskedConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn32   = nn.BatchNorm2d(32)

        self.conv33 = MaskedConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn33   = nn.BatchNorm2d(32)

        self.conv34 = MaskedConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn34   = nn.BatchNorm2d(32)

        self.conv35 = MaskedConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn35   = nn.BatchNorm2d(32)

        self.conv36 = MaskedConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn36   = nn.BatchNorm2d(32)

        self.conv37 = MaskedConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn37   = nn.BatchNorm2d(32)

        # ---
        self.conv38 = MaskedConv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn38   = nn.BatchNorm2d(64)

        self.conv39 = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn39   = nn.BatchNorm2d(64)

        self.conv40 = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn40   = nn.BatchNorm2d(64)

        self.conv41 = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn41   = nn.BatchNorm2d(64)

        self.conv42 = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn42   = nn.BatchNorm2d(64)

        self.conv43 = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn43   = nn.BatchNorm2d(64)

        self.conv44 = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn44   = nn.BatchNorm2d(64)

        self.conv45 = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn45   = nn.BatchNorm2d(64)

        self.conv46 = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn46   = nn.BatchNorm2d(64)

        self.conv47 = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn47   = nn.BatchNorm2d(64)

        self.conv48 = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn48   = nn.BatchNorm2d(64)

        self.conv49 = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn49   = nn.BatchNorm2d(64)

        self.conv50 = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn50   = nn.BatchNorm2d(64)

        self.conv51 = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn51   = nn.BatchNorm2d(64)

        self.conv52 = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn52   = nn.BatchNorm2d(64)

        self.conv53 = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn53   = nn.BatchNorm2d(64)

        self.conv54 = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn54   = nn.BatchNorm2d(64)

        self.conv55 = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn55   = nn.BatchNorm2d(64)
        
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
        #self.conv16.set_mask(torch.Tensor(masks['conv16.weight']))
        self.conv17.set_mask(torch.Tensor(masks['conv17.weight']))
        self.conv18.set_mask(torch.Tensor(masks['conv18.weight']))
        self.conv19.set_mask(torch.Tensor(masks['conv19.weight']))

        #self.conv20.set_mask(torch.Tensor(masks['conv20.weight']))
        self.conv21.set_mask(torch.Tensor(masks['conv21.weight']))
        self.conv22.set_mask(torch.Tensor(masks['conv22.weight']))
        self.conv23.set_mask(torch.Tensor(masks['conv23.weight']))
        self.conv24.set_mask(torch.Tensor(masks['conv24.weight']))
        self.conv25.set_mask(torch.Tensor(masks['conv25.weight']))
        self.conv26.set_mask(torch.Tensor(masks['conv26.weight']))
        self.conv27.set_mask(torch.Tensor(masks['conv27.weight']))
        self.conv28.set_mask(torch.Tensor(masks['conv28.weight']))
        self.conv29.set_mask(torch.Tensor(masks['conv29.weight']))

        self.conv30.set_mask(torch.Tensor(masks['conv30.weight']))
        self.conv31.set_mask(torch.Tensor(masks['conv31.weight']))
        self.conv32.set_mask(torch.Tensor(masks['conv32.weight']))
        self.conv33.set_mask(torch.Tensor(masks['conv33.weight']))
        self.conv34.set_mask(torch.Tensor(masks['conv34.weight']))
        self.conv35.set_mask(torch.Tensor(masks['conv35.weight']))
        self.conv36.set_mask(torch.Tensor(masks['conv36.weight']))
        self.conv37.set_mask(torch.Tensor(masks['conv37.weight']))
        #self.conv38.set_mask(torch.Tensor(masks['conv38.weight']))
        self.conv39.set_mask(torch.Tensor(masks['conv39.weight']))

        self.conv40.set_mask(torch.Tensor(masks['conv40.weight']))
        self.conv41.set_mask(torch.Tensor(masks['conv41.weight']))
        self.conv42.set_mask(torch.Tensor(masks['conv42.weight']))
        self.conv43.set_mask(torch.Tensor(masks['conv43.weight']))
        self.conv44.set_mask(torch.Tensor(masks['conv44.weight']))
        self.conv45.set_mask(torch.Tensor(masks['conv45.weight']))
        self.conv46.set_mask(torch.Tensor(masks['conv46.weight']))
        self.conv47.set_mask(torch.Tensor(masks['conv47.weight']))
        self.conv48.set_mask(torch.Tensor(masks['conv48.weight']))
        self.conv49.set_mask(torch.Tensor(masks['conv49.weight']))

        self.conv50.set_mask(torch.Tensor(masks['conv50.weight']))
        self.conv51.set_mask(torch.Tensor(masks['conv51.weight']))
        self.conv52.set_mask(torch.Tensor(masks['conv52.weight']))
        self.conv53.set_mask(torch.Tensor(masks['conv53.weight']))
        #self.conv54.set_mask(torch.Tensor(masks['conv54.weight']))
        self.conv55.set_mask(torch.Tensor(masks['conv55.weight']))
        
        #self.linear1.set_mask(torch.Tensor(masks['linear1.weight']))

    def forward(self, x, labels=False):
        
        out = self.relu(self.bn1(self.conv1(x)))
        
        # ----
        outer = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(outer)) + out) 
        outer = self.relu(self.bn4(self.conv4(out)))
        out = self.relu(self.bn5(self.conv5(outer)) + out) 
        outer = self.relu(self.bn6(self.conv6(out)))
        out = self.relu(self.bn7(self.conv7(outer)) + out) 
        outer = self.relu(self.bn8(self.conv8(out)))
        out = self.relu(self.bn9(self.conv9(outer)) + out) 
        outer = self.relu(self.bn10(self.conv10(out)))
        out = self.relu(self.bn11(self.conv11(outer)) + out) 
        outer = self.relu(self.bn12(self.conv12(out)))
        out = self.relu(self.bn13(self.conv13(outer)) + out) 
        outer = self.relu(self.bn14(self.conv14(out)))
        out = self.relu(self.bn15(self.conv15(outer)) + out) 
        outer = self.relu(self.bn16(self.conv16(out)))
        out = self.relu(self.bn17(self.conv17(outer)) + out) 
        outer = self.relu(self.bn18(self.conv18(out)))
        out = self.relu(self.bn19(self.conv19(outer)) + out) 

        # ----
        outer = self.relu(self.bn20(self.conv20(out)))
        out = self.relu(self.bn21(self.conv21(outer)) + F.pad(out[:, :, ::2, ::2], (0, 0, 0, 0, 32//4, 32//4), "constant", 0)) 
        
        outer = self.relu(self.bn22(self.conv22(out)))
        out = self.relu(self.bn23(self.conv23(outer)) + out) 
        outer = self.relu(self.bn24(self.conv24(out)))
        out = self.relu(self.bn25(self.conv25(outer)) + out) 
        outer = self.relu(self.bn26(self.conv26(out)))
        out = self.relu(self.bn27(self.conv27(outer)) + out) 
        outer = self.relu(self.bn28(self.conv28(out)))
        out = self.relu(self.bn29(self.conv29(outer)) + out) 
        outer = self.relu(self.bn30(self.conv30(out)))
        out = self.relu(self.bn31(self.conv31(outer)) + out) 
        outer = self.relu(self.bn32(self.conv32(out)))
        out = self.relu(self.bn33(self.conv33(outer)) + out) 
        outer = self.relu(self.bn34(self.conv34(out)))
        out = self.relu(self.bn35(self.conv35(outer)) + out) 
        outer = self.relu(self.bn36(self.conv36(out)))
        out = self.relu(self.bn37(self.conv37(outer)) + out) 

        # ----
        outer = self.relu(self.bn38(self.conv38(out)))
        out = self.relu(self.bn39(self.conv39(outer)) + F.pad(out[:, :, ::2, ::2], (0, 0, 0, 0, 64//4, 64//4), "constant", 0)) 
        
        outer = self.relu(self.bn40(self.conv40(out)))
        out = self.relu(self.bn41(self.conv41(outer)) + out) 
        outer = self.relu(self.bn42(self.conv42(out)))
        out = self.relu(self.bn43(self.conv43(outer)) + out) 
        outer = self.relu(self.bn44(self.conv44(out)))
        out = self.relu(self.bn45(self.conv45(outer)) + out) 
        outer = self.relu(self.bn46(self.conv46(out)))
        out = self.relu(self.bn47(self.conv47(outer)) + out) 
        outer = self.relu(self.bn48(self.conv48(out)))
        out = self.relu(self.bn49(self.conv49(outer)) + out) 
        outer = self.relu(self.bn50(self.conv50(out)))
        out = self.relu(self.bn51(self.conv51(outer)) + out) 
        outer = self.relu(self.bn52(self.conv52(out)))
        out = self.relu(self.bn53(self.conv53(outer)) + out) 
        outer = self.relu(self.bn54(self.conv54(out)))
        out = self.relu(self.bn55(self.conv55(outer)) + out) 


        # ----
        out = F.avg_pool2d(out, out.size()[3])

        out = out.view(out.size(0), -1)
        out = self.linear1(out)

        if labels:
            out = F.softmax(out, dim=1)

        return out

if __name__=="__main__":
    net = Resnet56(num_classes=10)
    out = net(torch.randn(2,3,32,32))
