import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class L2Net(nn.Module):

  def __init__(self, params):
    super(L2Net, self).__init__()
    # extract params
    input_depth = params["input_depth"]
    num_init_features = params["num_init_features"]
    do_batch_norm = params["do_batch_norm"]
    # setup network
    self.conv1 = torch.nn.Conv2d(input_depth, num_init_features, 3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(num_init_features, affine=do_batch_norm)

    self.conv2 = torch.nn.Conv2d(num_init_features, num_init_features, 3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(num_init_features, affine=do_batch_norm)

    self.conv3 = torch.nn.Conv2d(num_init_features, 2*num_init_features, 3, stride=1, padding=1)
    self.bn3 = nn.BatchNorm2d(2*num_init_features, affine=do_batch_norm)

    self.conv4 = torch.nn.Conv2d(2*num_init_features, 2*num_init_features, 3, stride=1, padding=1)
    self.bn4 = nn.BatchNorm2d(2*num_init_features, affine=do_batch_norm)

    self.conv5 = torch.nn.Conv2d(2*num_init_features, 4*num_init_features, 3, stride=1, padding=1)
    self.bn5 = nn.BatchNorm2d(4*num_init_features, affine=do_batch_norm)

    self.conv6 = torch.nn.Conv2d(4*num_init_features, 4*num_init_features, 3, stride=1, padding=1)
    self.bn6 = nn.BatchNorm2d(4*num_init_features, affine=do_batch_norm)

    self.conv7 = torch.nn.Conv2d(4*num_init_features, 1, 1)
    self.bn7 = nn.BatchNorm2d(1, affine=do_batch_norm)

    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.relu(self.bn2(self.conv2(x)))
    x = self.relu(self.bn3(self.conv3(x)))
    x = self.relu(self.bn4(self.conv4(x)))
    x = self.relu(self.bn5(self.conv5(x)))
    x = self.relu(self.bn6(self.conv6(x)))
    out = self.bn7(self.conv7(x))
    return out
