import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class DenseTransposeLayer(nn.Module):
  def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
    super(DenseTransposeLayer, self).__init__()
    self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
    self.add_module('relu1', nn.ReLU(inplace=True)),
    self.add_module('convT1', nn.ConvTranspose2d(num_input_features, bn_size *
      growth_rate, kernel_size=1, stride=1, padding=0,
      bias=False)),
    self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
    self.add_module('relu2', nn.ReLU(inplace=True)),
    self.add_module('convT2', nn.ConvTranspose2d(bn_size * growth_rate, growth_rate,
      kernel_size=3, stride=1, padding=1,
      bias=False)),
    self.drop_rate = float(drop_rate)

  def bn_function(self, inputs):
    # type: (List[Tensor]) -> Tensor
      concated_features = torch.cat(inputs, 1)
      bottleneck_output = self.convT1(self.relu1(self.norm1(concated_features)))  # noqa: T484
      return bottleneck_output

  def any_requires_grad(self, input):
    # type: (List[Tensor]) -> bool
    for tensor in input:
      if tensor.requires_grad:
        return True
    return False

  def forward(self, inputs):  # noqa: F811
    bottleneck_output = self.bn_function(inputs)
    new_features = self.convT2(self.relu2(self.norm2(bottleneck_output)))
    if self.drop_rate > 0:
      new_features = F.dropout(new_features, p=self.drop_rate,
          training=self.training)
    return new_features

class DenseTransposeBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(DenseTransposeBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseTransposeLayer(
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

class TransitionTransposeBlock(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(TransitionTransposeBlock, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('convT', nn.ConvTranspose2d(num_input_features, num_output_features,
                                          kernel_size=[1,4], stride=[1,2], padding=[0,1], bias=False))

class DenseDecoder(nn.Module):

  def __init__(self, params, input_depth):
    super(DenseDecoder, self).__init__()
    growth_rate = params["growth_rate"]
    block_configuration = params["block_configuration"]
    bn_size = params["bn_size"]
    drop_rate = params["drop_rate"]
    self.features = nn.Sequential()
    # self.features = nn.Sequential(OrderedDict([
    #   ('convT0', nn.ConvTranspose2d(input_depth, num_init_features, kernel_size=7, stride=[1,2],
    #     padding=3, bias=False)),
    #   ('norm0', nn.BatchNorm2d(num_init_features)),
    #   ('relu0', nn.ReLU(inplace=True)),
    #   ]))

    num_features = input_depth
    for i, block_size in enumerate(block_configuration):
      densetransposeblock = DenseTransposeBlock(block_size, num_features, bn_size, growth_rate, drop_rate)
      self.features.add_module('densetransposeblock%d' % (i + 1), densetransposeblock)

      num_features = num_features + block_size * growth_rate
      if i != len(block_configuration)-1:
        transitionblock = TransitionTransposeBlock(num_input_features=num_features, 
            num_output_features=num_features//2)
        self.features.add_module("transitiontransposeblock%d" % (i+1), transitionblock)
        num_features = num_features//2

    self.last_depth = num_features
    self.features.add_module("bn_decoder", nn.BatchNorm2d(num_features))
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
