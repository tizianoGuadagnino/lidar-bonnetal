import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class DenseLayer(nn.Module):
  def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
    super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
          growth_rate, kernel_size=1, stride=1, padding=0,
          bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
          kernel_size=3, stride=1, padding=1,
          bias=False)),
        self.drop_rate = float(drop_rate)

    def bn_function(self, inputs):
      # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    def any_requires_grad(self, input):
      # type: (List[Tensor]) -> bool
      for tensor in input:
        if tensor.requires_grad:
          return True
      return False

    def forward(self, input):  # noqa: F811
      bottleneck_output = self.bn_function(prev_features)
      new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
      if self.drop_rate > 0:
        new_features = F.dropout(new_features, p=self.drop_rate,
            training=self.training)
        return new_features

class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class TransitionBlock(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, padding=0, bias=False))
        # avg pooling that preserv the vertial dimension
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=[1,2], padding=[1,0]))

class DenseEncoder(nn.Module):

  def __init__(self, growth_rate=8, block_configuration=(3,3,3), input_depth=5, num_init_features=32, bn_size=4, drop_rate=0.0):
    self.features = nn.Sequential(OrderedDict([
      ('conv0', nn.Conv2d(input_depth, num_init_features, kernel_size=7, stride=[1,2],
        padding=3, bias=False)),
      ('norm0', nn.BatchNorm2d(num_init_features)),
      ('relu0', nn.ReLU(inplace=True)),
      ('pool0', nn.MaxPool2d(kernel_size=2, stride=[1,2], padding=[1,0])),
      ]))
      
      num_features = num_init_features
      for i, block_size in enumerate(block_configuration):
        denseblock = DenseBlock(block_size, num_features, bn_size, growth_rate, drop_rate)
        self.features.add_module('denseblock%d' % (i + 1), denseblock)
        
        num_features = num_features + block_size * growth_rate
        if i != len(block_configuration)-1:
          transitionblock = TransitionBlock(num_input_features=num_features, 
                                            num_output_features=num_input_features//2)
          self.features.add_module("transitionblock%d" % (i+1), transitionblock)
          num_features = num_features//2

      self.last_depth = num_features
      self.features.add_module("finalbatchnorm", nn.BatchNorm2d(num_features))
      for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

  def forward(self, x):
    preout = self.features(x)
    out = F.relu(preout, inplace=True)
    return out
