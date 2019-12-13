import torch 
import torch.nn as nn

class myNet(nn.Module):
  def __init__(self):
    super(myNet, self).__init__()
    self.conv = nn.Conv2d(3,10,2, stride = 2)
    self.relu = nn.ReLU()
    self.flatten = lambda x: x.view(-1)
    self.fc1 = nn.Linear(160,5)
    #self.seq = nn.Sequential(nn.Linear(5,3), nn.Linear(3,2))
    self.seq1 = nn.Linear(5,3)
    self.seq2 = nn.Linear(3,2)    
   
  
  def forward(self, x):
    x = self.relu(self.conv(x))
    x = x.view(-1,10*4*4)
    x = self.fc1(x)
    x = self.seq1(x)
    x = self.seq2(x)
    
  

net = myNet()
visualisation = {}
convert_code  = {}

def hook_fn(m, i, o):
  visualisation[str(m)] = o 

def get_all_layers(net):
  for name, layer in net._modules.items():
    print(name)
    #If it is a sequential, don't register a hook on it
    # but recursively register hook on all it's module children
    if isinstance(layer, nn.Sequential):
      get_all_layers(layer)
    else:
      # it's a non sequential. Register a hook
      layer.register_forward_hook(hook_fn)
      convert_code[name] = str(layer)

get_all_layers(net)

  
out = net(torch.randn(2,3,8,8))
import pdb; pdb.set_trace()

# Just to check whether we got all layers
print visualisation.keys()      #output includes sequential layers
