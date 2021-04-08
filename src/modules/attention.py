import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class AddNormAndReduce(nn.Module):
    def __init__(self, input_features, output_features):
        super(AddNormAndReduce).__init__()
        self.this = nn.Sequential()
        self.this.add_module('layer_norm', nn.LayerNorm(normalized_shape=input_features))
        self.this.add_module('conv', nn.Conv2d(input_features, output_features, kernel_size=1, stride=1, padding=0, bias=False))
        self.this.add_module('relu', nn.ReLU(inplace=True))
    def forward(self, x, y):
        out = x + y
        out = self.this(out)
        return out

class LidarAttention(nn.Module):
    def __init__(self, input_depth, num_features):
        super(LidarAttention, self).__init__()

        self.scale = nn.Parameter(torch.randn(1), requires_grad=True)

        self.query = nn.Sequential()
        self.query.add_module('conv_query', nn.Conv2d(input_depth,num_features,kernel_size=1, stride=1, padding=0, bias=False))
        self.query.add_module('bn_query', BatchNorm2d(num_features))
        self.query.add_module('relu_query', nn.ReLU(inplace=True))

        self.key = nn.Sequential()
        self.key.add_module('conv_key', nn.Conv2d(input_depth,num_features,kernel_size=1, stride=1, padding=0, bias=False))
        self.key.add_module('bn_key', BatchNorm2d(num_features))
        self.key.add_module('relu_key', nn.ReLU(inplace=True))

        self.value = nn.Sequential()
        self.value.add_module('conv_value', nn.Conv2d(input_depth,num_features,kernel_size=1, stride=1, padding=0, bias=False))
        self.value.add_module('bn_value', BatchNorm2d(num_features))
        self.value.add_module('relu_value', nn.ReLU(inplace=True))

        self.softmax = nn.Softmax(dim=1)
        self.residual_connection = AddNormAndReduce(num_features, num_features)

    def forward(self, q, k, v):
        yq = self.query(q)
        yk = self.key(k)
        yv = self.value(v)
        raw_weights = self.scale * yq * yk
        normalized_weights = self.softmax(raw_weights)
        attention = normalized_weights * yv
        out = self.residual_connection(attention, v)
        return out

class BasicConvBlock(nn.Module):
  def __init__(self, inplanes, planes):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,
                           stride=1, padding=0, bias=False)
    self.bn1 = nn.BatchNorm2d(planes[0])
    self.relu1 = nn.LeakyReLU(0.1)
    self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                           stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes[1])
    self.relu2 = nn.LeakyReLU(0.1)

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu1(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu2(out)

    out += residual
    return out
class EncoderAttentionBlock(nn.Module):
    def __init__(self, input_depth, inter_features, output_depth):
        super(EncoderAttentionBlock, self).__init__()
        
        self.this = nn.Sequential()
        self.this.add_module('attention1', LidarAttention(input_depth, inter_features))
        self.this.add_module('conv', BasicConvBlock(inter_features,))
        
