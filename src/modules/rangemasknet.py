#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from modules.encoder import *
from modules.decoder import *

class RangeMaskNet(nn.Module):
  def __init__(self, ARCH, path=None, path_append=""):
    super(RangeMaskNet, self).__init__()
    self.ARCH = ARCH
    encoder_params = self.ARCH["backbone"]["params"]
    self.backbone = L2Net(encoder_params)
    self.activation = nn.Sigmoid()
    # train backbone?
    if not self.ARCH["backbone"]["train"]:
      for w in self.backbone.parameters():
        w.requires_grad = False

    # print number of parameters and the ones requiring gradients
    # print number of parameters and the ones requiring gradients
    weights_total = sum(p.numel() for p in self.parameters())
    weights_grad = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print("Total number of parameters: ", weights_total)
    print("Total number of parameters requires_grad: ", weights_grad)
    # get weights
    if path is not None:
      # try backbone
      try:
        w_dict = torch.load(path + "/backbone" + path_append,
                            map_location=lambda storage, loc: storage)
        self.backbone.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model backbone weights")
      except Exception as e:
        print()
        print("Couldn't load backbone, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e
    else:
      print("No path to pretrained, using random init.")

  def forward(self, x, mask=None):
    # y = torch.cat(x,1)
    y = self.backbone(x)
    y = self.activation(y)
    # y = mask * y
    return y

  def save_checkpoint(self, logdir, suffix=""):
    # Save the weights
    torch.save(self.backbone.state_dict(), logdir +
               "/backbone" + suffix)
