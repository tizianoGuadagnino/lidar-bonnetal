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
    decoder_params = self.ARCH["decoder"]["params"]
    self.backbone = DenseEncoder(encoder_params)
    self.decoder = DenseDecoder(decoder_params, self.backbone.last_depth)
    self.head = nn.Sequential(nn.Dropout2d(p=self.ARCH["head"]["drop_rate"]),
                              nn.Conv2d(self.decoder.last_depth, 1, kernel_size=3, stride=1, padding=1))
    self.activation = nn.Sigmoid()
    # train backbone?
    if not self.ARCH["backbone"]["train"]:
      for w in self.backbone.parameters():
        w.requires_grad = False

    # train decoder?
    if not self.ARCH["decoder"]["train"]:
      for w in self.decoder.parameters():
        w.requires_grad = False

    # train head?
    if not self.ARCH["head"]["train"]:
      for w in self.head.parameters():
        w.requires_grad = False

    # print number of parameters and the ones requiring gradients
    # print number of parameters and the ones requiring gradients
    weights_total = sum(p.numel() for p in self.parameters())
    weights_grad = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print("Total number of parameters: ", weights_total)
    print("Total number of parameters requires_grad: ", weights_grad)

    # breakdown by layer
    weights_enc = sum(p.numel() for p in self.backbone.parameters())
    weights_dec = sum(p.numel() for p in self.decoder.parameters())
    weights_head = sum(p.numel() for p in self.head.parameters())
    print("Param encoder ", weights_enc)
    print("Param decoder ", weights_dec)
    print("Param head ", weights_head)
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

      # try decoder
      try:
        w_dict = torch.load(path + "/decoder" + path_append,
                            map_location=lambda storage, loc: storage)
        self.decoder.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model decoder weights")
      except Exception as e:
        print("Couldn't load decoder, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e

      # try head
      try:
        w_dict = torch.load(path + "/head" + path_append,
                            map_location=lambda storage, loc: storage)
        self.head.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model head weights")
      except Exception as e:
        print("Couldn't load head, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e
    else:
      print("No path to pretrained, using random init.")

  def forward(self, x, mask=None):
    # y = x[1] - x[0]
    y = torch.cat(x,1)
    y = self.backbone(y)
    y = self.decoder(y)
    y = self.head(y)
    y = self.activation(y)
    y = mask * y
    return y

  def save_checkpoint(self, logdir, suffix=""):
    # Save the weights
    torch.save(self.backbone.state_dict(), logdir +
               "/backbone" + suffix)
    torch.save(self.decoder.state_dict(), logdir +
               "/decoder" + suffix)
    torch.save(self.head.state_dict(), logdir +
               "/head" + suffix)
