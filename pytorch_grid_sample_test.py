import torch
from torch.nn import functional as F

batch = torch.randn(16,3,100,100)

def build_grid(source_size,target_size):
    k = float(target_size)/float(source_size)
    direct = torch.linspace(0,k,target_size).unsqueeze(0).repeat(target_size,1).unsqueeze(-1)
    full = torch.cat([direct,direct.transpose(1,0)],dim=2).unsqueeze(0)
    return full

def random_crop_grid(x,grid):
    delta = x.size(2)-grid.size(1)
    grid = grid.repeat(x.size(0),1,1,1)
    #Add random shifts by x
    grid[:,:,:,0] = grid[:,:,:,0]+ torch.FloatTensor(x.size(0)).random_(0, delta).unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2)) /x.size(2)
    #Add random shifts by y
    grid[:,:,:,1] = grid[:,:,:,1]+ torch.FloatTensor(x.size(0)).random_(0, delta).unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2)) /x.size(2)
    return grid
    
grid_source = build_grid(batch.size(2),80)

grid_shifted = random_crop_grid(batch,grid_source)

sampled_batch = F.grid_sample(batch, grid_shifted)