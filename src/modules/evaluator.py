#!/usr/bin/env python3

# This file is covered by the LICENSE file in the root of this project.

import torch
import numpy as np

class Evaluator:
  def __init__(self, device):
    self.device = device
    self.reset()

  def reset(self):
    self.conf_matrix = torch.zeros((2,2), device=self.device).long()
    self.ones = None
    self.last_scan_size = None  # for when variable scan size is used

  def addBatch(self, x, y):  # x=preds, y=targets
    # if numpy, pass to pytorch
    # to tensor
    if isinstance(x, np.ndarray):
      x = torch.from_numpy(np.array(x)).long().to(self.device)
    if isinstance(y, np.ndarray):
      y = torch.from_numpy(np.array(y)).long().to(self.device)

    long_x = x.long()
    long_y = y.long()
    # sizes should be "batch_size x H x W"
    x_row = long_x.reshape(-1)  # de-batchify
    y_row = long_y.reshape(-1)  # de-batchify

    # print("Positive in labels ", torch.count_nonzero(y_row))
    comp = (x_row == y_row).long()

    # count true positives
    self.conf_matrix[0,0] += torch.count_nonzero(comp[y_row == 1])
    # count true negatives
    self.conf_matrix[1,1] += torch.count_nonzero(comp[y_row == 0])
    # count false positives
    self.conf_matrix[0,1] += torch.count_nonzero(comp[y_row == 0] == 0)
    # count false negatives
    self.conf_matrix[1,0] +=  torch.count_nonzero(comp[y_row == 1] == 0)

  def getStats(self):
    # remove fp and fn from confusion on the ignore classes cols and rows
    conf = self.conf_matrix.clone().double()
    tp = conf[0,0]
    fp = conf[0,1]
    fn = conf[1,0]
    tn = conf[1,1]
    return tp, fp, fn, tn

  def getScores(self):
    tp, fp, fn, _ = self.getStats()
    precision = tp/(tp + fp + 1e-15)
    recall = tp/(tp + fn + 1e-15)
    f1 = 2 * (precision * recall) /(precision + recall + 1e-15)
    return precision, recall, f1

  def getacc(self):
    tp, fp, fn, tn = self.getStats()
    cc = tp + tn
    total = tp + tn + fp + fn
    acc_mean = cc / total
    return acc_mean  # returns "acc mean"


if __name__ == "__main__":
  # mock problem
  nclasses = 2
  ignore = []

  # test with 2 squares and a known IOU
  lbl = torch.zeros((7, 7)).long()
  argmax = torch.zeros((7, 7)).long()

  # put squares
  lbl[2:4, 2:4] = 1
  argmax[3:5, 3:5] = 1

  # make evaluator
  eval = iouEval(nclasses, torch.device('cpu'), ignore)

  # run
  eval.addBatch(argmax, lbl)
  m_iou, iou = eval.getIoU()
  print("*"*80)
  print("Small iou mock problem")
  print("IoU: ", m_iou)
  print("IoU class: ", iou)
  m_acc = eval.getacc()
  print("Acc: ", m_acc)
  print("*"*80)
