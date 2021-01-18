import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, pos_weight: Optional[Tensor] = None, gamma: Optional[Tensor]=None):
        super(FocalLoss).__init__()
        self.w = pos_weight
        self.gamma = gamma
    def forward(self, input, target):
        bce = F.binary_cross_entropy(input, target, reduction=None)
        pt = F.exp(-bce)
        ones = torch.ones(target.shape,dtype=torch.float)
        focal_factor = torch.pow(ones-pt,self.gamma)
        W = torch.ones(target.shape,dtype=torch.float)
        W[target==1] = self.w
        pixelwise_loss = W * focal_factor * bce
        return torch.mean(pixelwise_loss)

