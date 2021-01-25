import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, alpha = 1.0, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, input, target, mask=None):
        loss = F.binary_cross_entropy(input, target, reduction='none')
        if mask is not None:
            bce = mask * loss
        else:
            bce = loss
        pt = torch.exp(-bce)
        focal_factor = (1.-pt)**self.gamma
        if self.alpha > 1:
            W = target * (self.alpha-1) + 1
            pixelwise_loss = W * focal_factor * bce
        else:
            pixelwise_loss = bce
        return pixelwise_loss.mean()
