import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


import torch
import torch.nn as nn
import torchvision.models as models

from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torch.autograd        import Variable
from .layers                import *


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}
class Resnet50(nn.Module):

    def __init__(self, num_classes=10):
        super(Resnet50, self).__init__()

        self.relu = nn.ReLU()

        self.conv1   = MaskedConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ---
        self.conv2  = MaskedConv2d(64, 64, kernel_size=1, stride=1, bias=False)
        self.bn2    = nn.BatchNorm2d(64)

        self.conv3  = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3    = nn.BatchNorm2d(64)

        self.conv4  = MaskedConv2d(64, 256, kernel_size=1, stride=1, bias=False)
        self.bn4    = nn.BatchNorm2d(256)

        self.conv5  = MaskedConv2d(64, 256, kernel_size=1, stride=1, bias=False)
        self.bn5    = nn.BatchNorm2d(256)

        # ---
        self.conv6  = MaskedConv2d(256, 64, kernel_size=1, stride=1, bias=False)
        self.bn6    = nn.BatchNorm2d(64)

        self.conv7  = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7    = nn.BatchNorm2d(64)

        self.conv8  = MaskedConv2d(64, 256, kernel_size=1, stride=1, bias=False)
        self.bn8    = nn.BatchNorm2d(256)

        # ---
        self.conv9  = MaskedConv2d(256, 64, kernel_size=1, stride=1, bias=False)
        self.bn9    = nn.BatchNorm2d(64)

        self.conv10 = MaskedConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn10   = nn.BatchNorm2d(64)

        self.conv11 = MaskedConv2d(64, 256, kernel_size=1, stride=1, bias=False)
        self.bn11   = nn.BatchNorm2d(256)

        # ---
        self.conv12 = MaskedConv2d(256, 128, kernel_size=1, stride=1, bias=False)
        self.bn12   = nn.BatchNorm2d(128)
        
        self.conv13 = MaskedConv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn13   = nn.BatchNorm2d(128)
        
        self.conv14 = MaskedConv2d(128, 512, kernel_size=1, stride=1, bias=False)
        self.bn14   = nn.BatchNorm2d(512)
        
        self.conv15 = MaskedConv2d(256, 512, kernel_size=1, stride=2, bias=False)
        self.bn15   = nn.BatchNorm2d(512)
        
        # ---
        self.conv16 = MaskedConv2d(512, 128, kernel_size=1, stride=1, bias=False)
        self.bn16   = nn.BatchNorm2d(128)
        
        self.conv17 = MaskedConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn17   = nn.BatchNorm2d(128)
        
        self.conv18 = MaskedConv2d(128, 512, kernel_size=1, stride=1, bias=False)
        self.bn18   = nn.BatchNorm2d(512)
        
        # --- 
        self.conv19 = MaskedConv2d(512, 128, kernel_size=1, stride=1, bias=False)
        self.bn19   = nn.BatchNorm2d(128)
        
        self.conv20 = MaskedConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn20   = nn.BatchNorm2d(128)

        self.conv21 = MaskedConv2d(128, 512, kernel_size=1, stride=1, bias=False)
        self.bn21   = nn.BatchNorm2d(512)

        # ---
        self.conv22 = MaskedConv2d(512, 128, kernel_size=1, stride=1, bias=False)
        self.bn22   = nn.BatchNorm2d(128)

        self.conv23 = MaskedConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn23   = nn.BatchNorm2d(128)

        self.conv24 = MaskedConv2d(128, 512, kernel_size=1, stride=1, bias=False)
        self.bn24   = nn.BatchNorm2d(512)

        # ---
        self.conv25 = MaskedConv2d(512, 256, kernel_size=1, stride=1, bias=False)
        self.bn25   = nn.BatchNorm2d(256)

        self.conv26 = MaskedConv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn26   = nn.BatchNorm2d(256)

        self.conv27 = MaskedConv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.bn27   = nn.BatchNorm2d(1024)

        self.conv28 = MaskedConv2d(512, 1024, kernel_size=1, stride=2, bias=False)
        self.bn28   = nn.BatchNorm2d(1024)
        
        # --- 
        self.conv29 = MaskedConv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.bn29   = nn.BatchNorm2d(256)

        self.conv30 = MaskedConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn30   = nn.BatchNorm2d(256)

        self.conv31 = MaskedConv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.bn31   = nn.BatchNorm2d(1024)

        # ---
        self.conv32 = MaskedConv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.bn32   = nn.BatchNorm2d(256)

        self.conv33 = MaskedConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn33   = nn.BatchNorm2d(256)

        self.conv34 = MaskedConv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.bn34   = nn.BatchNorm2d(1024)

        # --- 
        self.conv35 = MaskedConv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.bn35   = nn.BatchNorm2d(256)

        self.conv36 = MaskedConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn36   = nn.BatchNorm2d(256)

        self.conv37 = MaskedConv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.bn37   = nn.BatchNorm2d(1024)

        # ---
        self.conv38 = MaskedConv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.bn38   = nn.BatchNorm2d(256)

        self.conv39 = MaskedConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn39   = nn.BatchNorm2d(256)

        self.conv40 = MaskedConv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.bn40   = nn.BatchNorm2d(1024)

        # ---
        self.conv41 = MaskedConv2d(1024, 256, kernel_size=1, stride=1, bias=False)
        self.bn41   = nn.BatchNorm2d(256)

        self.conv42 = MaskedConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn42   = nn.BatchNorm2d(256)

        self.conv43 = MaskedConv2d(256, 1024, kernel_size=1, stride=1, bias=False)
        self.bn43   = nn.BatchNorm2d(1024)

        # ---
        self.conv44 = MaskedConv2d(1024, 512, kernel_size=1, stride=1, bias=False)
        self.bn44   = nn.BatchNorm2d(512)

        self.conv45 = MaskedConv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn45   = nn.BatchNorm2d(512)

        self.conv46 = MaskedConv2d(512, 2048, kernel_size=1, stride=1, bias=False)
        self.bn46   = nn.BatchNorm2d(2048)

        self.conv47 = MaskedConv2d(1024, 2048, kernel_size=1, stride=2, bias=False)
        self.bn47   = nn.BatchNorm2d(2048)

        # ---
        self.conv48 = MaskedConv2d(2048, 512, kernel_size=1, stride=1, bias=False)
        self.bn48   = nn.BatchNorm2d(512)

        self.conv49 = MaskedConv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn49   = nn.BatchNorm2d(512)

        self.conv50 = MaskedConv2d(512, 2048, kernel_size=1, stride=1, bias=False)
        self.bn50   = nn.BatchNorm2d(2048)

        # ---
        self.conv51 = MaskedConv2d(2048, 512, kernel_size=1, stride=1, bias=False)
        self.bn51   = nn.BatchNorm2d(512)

        self.conv52 = MaskedConv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn52   = nn.BatchNorm2d(512)

        self.conv53 = MaskedConv2d(512, 2048, kernel_size=1, stride=1, bias=False)
        self.bn53   = nn.BatchNorm2d(2048)

        # ---
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear1 = MaskedLinear(2048, num_classes)



    def setup_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        self.conv2.set_mask(torch.Tensor(masks['conv2.weight']).cuda())
        self.conv3.set_mask(torch.Tensor(masks['conv3.weight']).cuda())
        self.conv4.set_mask(torch.Tensor(masks['conv4.weight']).cuda())
        self.conv5.set_mask(torch.Tensor(masks['conv5.weight']).cuda())
        self.conv6.set_mask(torch.Tensor(masks['conv6.weight']).cuda())
        self.conv7.set_mask(torch.Tensor(masks['conv7.weight']).cuda())
        self.conv8.set_mask(torch.Tensor(masks['conv8.weight']).cuda())
        self.conv9.set_mask(torch.Tensor(masks['conv9.weight']).cuda())
        
        self.conv10.set_mask(torch.Tensor(masks['conv10.weight']).cuda())
        self.conv11.set_mask(torch.Tensor(masks['conv11.weight']).cuda())
        self.conv12.set_mask(torch.Tensor(masks['conv12.weight']).cuda())
        self.conv13.set_mask(torch.Tensor(masks['conv13.weight']).cuda())
        self.conv14.set_mask(torch.Tensor(masks['conv14.weight']).cuda())
        self.conv15.set_mask(torch.Tensor(masks['conv15.weight']).cuda())
        self.conv16.set_mask(torch.Tensor(masks['conv16.weight']).cuda())
        self.conv17.set_mask(torch.Tensor(masks['conv17.weight']).cuda())
        self.conv18.set_mask(torch.Tensor(masks['conv18.weight']).cuda())
        self.conv19.set_mask(torch.Tensor(masks['conv19.weight']).cuda())

        self.conv20.set_mask(torch.Tensor(masks['conv20.weight']).cuda())
        self.conv21.set_mask(torch.Tensor(masks['conv21.weight']).cuda())
        self.conv22.set_mask(torch.Tensor(masks['conv22.weight']).cuda())
        self.conv23.set_mask(torch.Tensor(masks['conv23.weight']).cuda())
        self.conv24.set_mask(torch.Tensor(masks['conv24.weight']).cuda())
        self.conv25.set_mask(torch.Tensor(masks['conv25.weight']).cuda())
        self.conv26.set_mask(torch.Tensor(masks['conv26.weight']).cuda())
        self.conv27.set_mask(torch.Tensor(masks['conv27.weight']).cuda())
        self.conv28.set_mask(torch.Tensor(masks['conv28.weight']).cuda())
        self.conv29.set_mask(torch.Tensor(masks['conv29.weight']).cuda())

        self.conv30.set_mask(torch.Tensor(masks['conv30.weight']).cuda())
        self.conv31.set_mask(torch.Tensor(masks['conv31.weight']).cuda())
        self.conv32.set_mask(torch.Tensor(masks['conv32.weight']).cuda())
        self.conv33.set_mask(torch.Tensor(masks['conv33.weight']).cuda())
        self.conv34.set_mask(torch.Tensor(masks['conv34.weight']).cuda())
        self.conv35.set_mask(torch.Tensor(masks['conv35.weight']).cuda())
        self.conv36.set_mask(torch.Tensor(masks['conv36.weight']).cuda())
        self.conv37.set_mask(torch.Tensor(masks['conv37.weight']).cuda())
        self.conv38.set_mask(torch.Tensor(masks['conv38.weight']).cuda())
        self.conv39.set_mask(torch.Tensor(masks['conv39.weight']).cuda())

        self.conv40.set_mask(torch.Tensor(masks['conv40.weight']).cuda())
        self.conv41.set_mask(torch.Tensor(masks['conv41.weight']).cuda())
        self.conv42.set_mask(torch.Tensor(masks['conv42.weight']).cuda())
        self.conv43.set_mask(torch.Tensor(masks['conv43.weight']).cuda())
        self.conv44.set_mask(torch.Tensor(masks['conv44.weight']).cuda())
        self.conv45.set_mask(torch.Tensor(masks['conv45.weight']).cuda())
        self.conv46.set_mask(torch.Tensor(masks['conv46.weight']).cuda())
        self.conv47.set_mask(torch.Tensor(masks['conv47.weight']).cuda())
        self.conv48.set_mask(torch.Tensor(masks['conv48.weight']).cuda())
        self.conv49.set_mask(torch.Tensor(masks['conv49.weight']).cuda())

        self.conv50.set_mask(torch.Tensor(masks['conv50.weight']).cuda())
        self.conv51.set_mask(torch.Tensor(masks['conv51.weight']).cuda())
        self.conv52.set_mask(torch.Tensor(masks['conv52.weight']).cuda())
        self.conv53.set_mask(torch.Tensor(masks['conv53.weight']).cuda())

    def forward(self, x, labels=False):
        
        outer = self.maxpool(self.relu(self.bn1(self.conv1(x))))
    
        # ----
        out = self.relu(self.bn2(self.conv2(outer)))
        out = self.relu(self.bn3(self.conv3(out))) 
        out = self.bn4(self.conv4(out))
        outer = self.relu(self.bn5(self.conv5(outer)) + out) 

        # ---
        out = self.relu(self.bn6(self.conv6(outer)))
        out = self.relu(self.bn7(self.conv7(out))) 
        outer = self.relu(self.bn8(self.conv8(out)) + outer)

        # ---
        out = self.relu(self.bn9(self.conv9(outer))) 
        out = self.relu(self.bn10(self.conv10(out)))
        outer = self.relu(self.bn11(self.conv11(out))+ outer) 

        # ---
        out = self.relu(self.bn12(self.conv12(outer)))
        out = self.relu(self.bn13(self.conv13(out))) 
        out = self.bn14(self.conv14(out))
        outer = self.relu(self.bn15(self.conv15(outer)) + out) 
        
        # ---
        out = self.relu(self.bn16(self.conv16(outer)))
        out = self.relu(self.bn17(self.conv17(out))) 
        outer = self.relu(self.bn18(self.conv18(out)) + outer)

        # ---
        out = self.relu(self.bn19(self.conv19(outer))) 
        out = self.relu(self.bn20(self.conv20(out)))
        outer = self.relu(self.bn21(self.conv21(out)) + outer) 
       
        # --- 
        out = self.relu(self.bn22(self.conv22(outer)))
        out = self.relu(self.bn23(self.conv23(out))) 
        outer = self.relu(self.bn24(self.conv24(out)) + outer)
            
        # ---
        out = self.relu(self.bn25(self.conv25(outer))) 
        out = self.relu(self.bn26(self.conv26(out)))
        out = self.bn27(self.conv27(out)) 
        outer = self.relu(self.bn28(self.conv28(outer)) + out)

        # ---
        out = self.relu(self.bn29(self.conv29(outer))) 
        out = self.relu(self.bn30(self.conv30(out)))
        outer = self.relu(self.bn31(self.conv31(out)) + outer) 
    
        # ---
        out = self.relu(self.bn32(self.conv32(outer)))
        out = self.relu(self.bn33(self.conv33(out))) 
        outer = self.relu(self.bn34(self.conv34(out)) + outer)

        # ---
        out = self.relu(self.bn35(self.conv35(outer))) 
        out = self.relu(self.bn36(self.conv36(out)))
        outer = self.relu(self.bn37(self.conv37(out)) + outer) 

        # ----
        out = self.relu(self.bn38(self.conv38(outer)))
        out = self.relu(self.bn39(self.conv39(out))) 
        outer = self.relu(self.bn40(self.conv40(out)) + outer)
    
        # ---
        out = self.relu(self.bn41(self.conv41(outer))) 
        out = self.relu(self.bn42(self.conv42(out)))
        outer = self.relu(self.bn43(self.conv43(out)) + outer) 

        # ---
        out = self.relu(self.bn44(self.conv44(outer)))
        out = self.relu(self.bn45(self.conv45(out))) 
        out = self.bn46(self.conv46(out))
        outer = self.relu(self.bn47(self.conv47(outer)) + out) 

        # ---
        out = self.relu(self.bn48(self.conv48(outer)))
        out = self.relu(self.bn49(self.conv49(out))) 
        outer = self.relu(self.bn50(self.conv50(out)) + outer)

        # ---
        out = self.relu(self.bn51(self.conv51(outer))) 
        out = self.relu(self.bn52(self.conv52(out)))
        out = self.relu(self.bn53(self.conv53(out)) + outer) 

        # ---
        out = self.avgpool(out) 

        out = torch.flatten(out,1)
        #out.view(out.size(0), -1)
        out = self.linear1(out)

        if labels:
            out = F.softmax(out, dim=1)

        return out

def resnet50(num_classes):
    model = Resnet50(num_classes)

    #state_dict = load_state_dict_from_url(model_urls['resnet50'],
    #                                      progress=True)
    #new_state_dict = {}
    #new_state_dict['conv1.weight'] = state_dict['conv1.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn1'+ite] = state_dict['bn1'+ite]

    #new_state_dict['conv2.weight'] = state_dict['layer1.0.conv1.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn2'+ite] = state_dict['layer1.0.bn1'+ite]

    #new_state_dict['conv3.weight'] = state_dict['layer1.0.conv2.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn3'+ite] = state_dict['layer1.0.bn2'+ite]

    #new_state_dict['conv4.weight'] = state_dict['layer1.0.conv3.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn4'+ite] = state_dict['layer1.0.bn3'+ite]

    #new_state_dict['conv5.weight'] = state_dict['layer1.0.downsample.0.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn5'+ite] = state_dict['layer1.0.downsample.1'+ite]

    #new_state_dict['conv6.weight'] = state_dict['layer1.1.conv1.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn6'+ite] = state_dict['layer1.1.bn1'+ite]

    #new_state_dict['conv7.weight'] = state_dict['layer1.1.conv2.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn7'+ite] = state_dict['layer1.1.bn2'+ite]

    #new_state_dict['conv8.weight'] = state_dict['layer1.1.conv3.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn8'+ite] = state_dict['layer1.1.bn3'+ite]

    #new_state_dict['conv9.weight'] = state_dict['layer1.2.conv1.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn9'+ite] = state_dict['layer1.2.bn1'+ite]

    #new_state_dict['conv10.weight'] = state_dict['layer1.2.conv2.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn10'+ite] = state_dict['layer1.2.bn2'+ite]

    #new_state_dict['conv11.weight'] = state_dict['layer1.2.conv3.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn11'+ite] = state_dict['layer1.2.bn3'+ite]


    #new_state_dict['conv12.weight'] = state_dict['layer2.0.conv1.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn12'+ite] = state_dict['layer2.0.bn1'+ite]

    #new_state_dict['conv13.weight'] = state_dict['layer2.0.conv2.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn13'+ite] = state_dict['layer2.0.bn2'+ite]

    #new_state_dict['conv14.weight'] = state_dict['layer2.0.conv3.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn14'+ite] = state_dict['layer2.0.bn3'+ite]

    #new_state_dict['conv15.weight'] = state_dict['layer2.0.downsample.0.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn15'+ite] = state_dict['layer2.0.downsample.1'+ite]

    #new_state_dict['conv16.weight'] = state_dict['layer2.1.conv1.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn16'+ite] = state_dict['layer2.1.bn1'+ite]

    #new_state_dict['conv17.weight'] = state_dict['layer2.1.conv2.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn17'+ite] = state_dict['layer2.1.bn2'+ite]

    #new_state_dict['conv18.weight'] = state_dict['layer2.1.conv3.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn18'+ite] = state_dict['layer2.1.bn3'+ite]

    #new_state_dict['conv19.weight'] = state_dict['layer2.2.conv1.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn19'+ite] = state_dict['layer2.2.bn1'+ite]

    #new_state_dict['conv20.weight'] = state_dict['layer2.2.conv2.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn20'+ite] = state_dict['layer2.2.bn2'+ite]

    #new_state_dict['conv21.weight'] = state_dict['layer2.2.conv3.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn21'+ite] = state_dict['layer2.2.bn3'+ite]

    #new_state_dict['conv22.weight'] = state_dict['layer2.3.conv1.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn22'+ite] = state_dict['layer2.3.bn1'+ite]

    #new_state_dict['conv23.weight'] = state_dict['layer2.3.conv2.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn23'+ite] = state_dict['layer2.3.bn2'+ite]

    #new_state_dict['conv24.weight'] = state_dict['layer2.3.conv3.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn24'+ite] = state_dict['layer2.3.bn3'+ite]

    #new_state_dict['conv25.weight'] = state_dict['layer3.0.conv1.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn25'+ite] = state_dict['layer3.0.bn1'+ite]

    #new_state_dict['conv26.weight'] = state_dict['layer3.0.conv2.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn26'+ite] = state_dict['layer3.0.bn2'+ite]

    #new_state_dict['conv27.weight'] = state_dict['layer3.0.conv3.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn27'+ite] = state_dict['layer3.0.bn3'+ite]

    #new_state_dict['conv28.weight'] = state_dict['layer3.0.downsample.0.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn28'+ite] = state_dict['layer3.0.downsample.1'+ite]

    #new_state_dict['conv29.weight'] = state_dict['layer3.1.conv1.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn29'+ite] = state_dict['layer3.1.bn1'+ite]

    #new_state_dict['conv30.weight'] = state_dict['layer3.1.conv2.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn30'+ite] = state_dict['layer3.1.bn2'+ite]

    #new_state_dict['conv31.weight'] = state_dict['layer3.1.conv3.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn31'+ite] = state_dict['layer3.1.bn3'+ite]

    #new_state_dict['conv32.weight'] = state_dict['layer3.2.conv1.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn32'+ite] = state_dict['layer3.2.bn1'+ite]

    #new_state_dict['conv33.weight'] = state_dict['layer3.2.conv2.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn33'+ite] = state_dict['layer3.2.bn2'+ite]

    #new_state_dict['conv34.weight'] = state_dict['layer3.2.conv3.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn34'+ite] = state_dict['layer3.2.bn3'+ite]

    #new_state_dict['conv35.weight'] = state_dict['layer3.3.conv1.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn35'+ite] = state_dict['layer3.3.bn1'+ite]

    #new_state_dict['conv36.weight'] = state_dict['layer3.3.conv2.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn36'+ite] = state_dict['layer3.3.bn2'+ite]

    #new_state_dict['conv37.weight'] = state_dict['layer3.3.conv3.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn37'+ite] = state_dict['layer3.3.bn3'+ite]

    #new_state_dict['conv38.weight'] = state_dict['layer3.4.conv1.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn38'+ite] = state_dict['layer3.4.bn1'+ite]

    #new_state_dict['conv39.weight'] = state_dict['layer3.4.conv2.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn39'+ite] = state_dict['layer3.4.bn2'+ite]

    #new_state_dict['conv40.weight'] = state_dict['layer3.4.conv3.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn40'+ite] = state_dict['layer3.4.bn3'+ite]

    #new_state_dict['conv41.weight'] = state_dict['layer3.5.conv1.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn41'+ite] = state_dict['layer3.5.bn1'+ite]

    #new_state_dict['conv42.weight'] = state_dict['layer3.5.conv2.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn42'+ite] = state_dict['layer3.5.bn2'+ite]

    #new_state_dict['conv43.weight'] = state_dict['layer3.5.conv3.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn43'+ite] = state_dict['layer3.5.bn3'+ite]


    #new_state_dict['conv44.weight'] = state_dict['layer4.0.conv1.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn44'+ite] = state_dict['layer4.0.bn1'+ite]

    #new_state_dict['conv45.weight'] = state_dict['layer4.0.conv2.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn45'+ite] = state_dict['layer4.0.bn2'+ite]

    #new_state_dict['conv46.weight'] = state_dict['layer4.0.conv3.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn46'+ite] = state_dict['layer4.0.bn3'+ite]

    #new_state_dict['conv47.weight'] = state_dict['layer4.0.downsample.0.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn47'+ite] = state_dict['layer4.0.downsample.1'+ite]

    #new_state_dict['conv48.weight'] = state_dict['layer4.1.conv1.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn48'+ite] = state_dict['layer4.1.bn1'+ite]

    #new_state_dict['conv49.weight'] = state_dict['layer4.1.conv2.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn49'+ite] = state_dict['layer4.1.bn2'+ite]

    #new_state_dict['conv50.weight'] = state_dict['layer4.1.conv3.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn50'+ite] = state_dict['layer4.1.bn3'+ite]

    #new_state_dict['conv51.weight'] = state_dict['layer4.2.conv1.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn51'+ite] = state_dict['layer4.2.bn1'+ite]

    #new_state_dict['conv52.weight'] = state_dict['layer4.2.conv2.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn52'+ite] = state_dict['layer4.2.bn2'+ite]

    #new_state_dict['conv53.weight'] = state_dict['layer4.2.conv3.weight']
    #for ite in ['.running_var', '.weight', '.running_mean','.bias']:
    #    new_state_dict['bn53'+ite] = state_dict['layer4.2.bn3'+ite]


    #import pdb; pdb.set_trace()
    #new_state_dict['linear1.weight'] = state_dict['fc.weight']
    #new_state_dict['linear1.bias']   = state_dict['fc.bias']


    #torch.save({'state_dict': new_state_dict}, 'results/BASELINE_IMAGENET2012_RESNET50/0/logits_best.pkl')     

    #model.load_state_dict(new_state_dict)

    return model

if __name__=="__main__":
    net = resnet50(num_classes=1000)
    import pdb; pdb.set_trace()

