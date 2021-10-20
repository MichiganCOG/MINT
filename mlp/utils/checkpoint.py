import os
import torch

#### Save Current Checkpoint Details ####
def save_checkpoint(epoch, step, model, optimizer, filename):
    state = {   'epoch':epoch,
                'step': step,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
             }

    path = os.path.join('results/', filename)

    torch.save(state, path)
   

#### Load state dict ####
def load_checkpoint(name):
    checkpoint = torch.load(os.path.join('results',name))

    return checkpoint["state_dict"]
