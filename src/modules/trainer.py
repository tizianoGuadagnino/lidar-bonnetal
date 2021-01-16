#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import imp
import yaml
import time
from PIL import Image
import __init__ as booger
import collections
import copy
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

from utils.logger import Logger
from utils.avgmeter import *
from utils.warmupLR import *
from modules.rangemasknet import *
from modules.evaluator import *


class Trainer():
  def __init__(self, ARCH, DATA, datadir, logdir, path=None):
    # parameters
    self.ARCH = ARCH
    self.DATA = DATA
    self.datadir = datadir
    self.log = logdir
    self.path = path

    # put logger where it belongs
    self.tb_logger = Logger(self.log + "/tb")
    self.info = {"train_update": 0,
                 "train_loss": 0,
                 "train_acc": 0,
                 "train_iou": 0,
                 "valid_loss": 0,
                 "valid_acc": 0,
                 "valid_iou": 0,
                 "backbone_lr": 0,
                 "decoder_lr": 0,
                 "head_lr": 0,
                 "post_lr": 0}

    # get the data
    parserModule = imp.load_source("parserModule",
                                   booger.TRAIN_PATH + "dataset/" + self.DATA["name"] + '/parser.py')
    self.parser = parserModule.Parser(root=self.datadir,
                                      train_sequences=self.DATA["split"]["train"],
                                      valid_sequences=self.DATA["split"]["valid"],
                                      test_sequences=None,
                                      sensor=self.ARCH["dataset"]["sensor"],
                                      max_points=self.ARCH["dataset"]["max_points"],
                                      batch_size=self.ARCH["train"]["batch_size"],
                                      workers=self.ARCH["train"]["workers"],
                                      gt=True,
                                      shuffle_train=True)

    # concatenate the encoder and the head
    with torch.no_grad():
      self.model = RangeMaskNet(self.ARCH,
                               self.path)

    # GPU?
    self.gpu = False
    self.multi_gpu = False
    self.n_gpus = 0
    self.model_single = self.model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training in device: ", self.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      cudnn.benchmark = True
      cudnn.fastest = True
      self.gpu = True
      self.n_gpus = 1
      self.model.cuda()
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      self.model = nn.DataParallel(self.model)   # spread in gpus
      self.model = convert_model(self.model).cuda()  # sync batchnorm
      self.model_single = self.model.module  # single model to get weight names
      self.multi_gpu = True
      self.n_gpus = torch.cuda.device_count()

    # loss
    # w = torch.zeros(1,dtype=torch.float)
    # w[0] = 1e2
    self.criterion = nn.BCELoss().to(self.device)
    # loss as dataparallel too (more images in batch)
    if self.n_gpus > 1:
      self.criterion = nn.DataParallel(self.criterion).cuda()  # spread in gpus

    # optimizer
    self.lr_group_names = []
    self.train_dicts = []
    if self.ARCH["backbone"]["train"]:
      self.lr_group_names.append("backbone_lr")
      self.train_dicts.append(
          {'params': self.model_single.backbone.parameters()})
    if self.ARCH["decoder"]["train"]:
      self.lr_group_names.append("decoder_lr")
      self.train_dicts.append(
          {'params': self.model_single.decoder.parameters()})
    if self.ARCH["head"]["train"]:
      self.lr_group_names.append("head_lr")
      self.train_dicts.append({'params': self.model_single.head.parameters()})

    # Use SGD optimizer to train
    self.optimizer = optim.SGD(self.train_dicts,
                               lr=self.ARCH["train"]["lr"],
                               momentum=self.ARCH["train"]["momentum"],
                               weight_decay=self.ARCH["train"]["w_decay"])

    # Use warmup learning rate
    # post decay and step sizes come in epochs and we want it in steps
    steps_per_epoch = self.parser.get_train_size()
    up_steps = int(self.ARCH["train"]["wup_epochs"] * steps_per_epoch)
    final_decay = self.ARCH["train"]["lr_decay"] ** (1/steps_per_epoch)
    self.scheduler = warmupLR(optimizer=self.optimizer,
                              lr=self.ARCH["train"]["lr"],
                              warmup_steps=up_steps,
                              momentum=self.ARCH["train"]["momentum"],
                              decay=final_decay)

  @staticmethod
  def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)
    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)
    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
    return color_range.reshape(256, 1, 3)

  @staticmethod
  def make_log_img(depth, mask, pred, gt, color_fn):
    # input should be [depth, pred, gt]
    # make range image (normalized to 0,1 for saving)
    depth = (cv2.normalize(depth, None, alpha=0, beta=1,
                           norm_type=cv2.NORM_MINMAX,
                           dtype=cv2.CV_32F) * 255.0).astype(np.uint8)
    out_img = cv2.applyColorMap(
        depth, Trainer.get_mpl_colormap('viridis')) * mask[..., None]
    # make label prediction
    pred_color = color_fn((pred * mask).astype(np.int32))
    out_img = np.concatenate([out_img, pred_color], axis=0)
    # make label gt
    gt_color = color_fn(gt)
    out_img = np.concatenate([out_img, gt_color], axis=0)
    return (out_img).astype(np.uint8)

  @staticmethod
  def save_to_log(logdir, logger, info, epoch, w_summary=False, model=None, img_summary=False, imgs=[]):
    # save scalars
    for tag, value in info.items():
      logger.scalar_summary(tag, value, epoch)

    # save summaries of weights and biases
    if w_summary and model:
      for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
        if value.grad is not None:
          logger.histo_summary(
              tag + '/grad', value.grad.data.cpu().numpy(), epoch)

    if img_summary and len(imgs) > 0:
      directory = os.path.join(logdir, "predictions")
      if not os.path.isdir(directory):
        os.makedirs(directory)
      for i, img in enumerate(imgs):
        name = os.path.join(directory, str(i) + ".png")
        cv2.imwrite(name, img)

  def train(self):
    # best validation loss so far
    best_val_loss = 0

    self.evaluator = Evaluator(self.device)
    # train for n epochs
    for epoch in range(self.ARCH["train"]["max_epochs"]):
      # get info for learn rate currently
      groups = self.optimizer.param_groups
      for name, g in zip(self.lr_group_names, groups):
        self.info[name] = g['lr']

      # train for 1 epoch
      loss, update_mean = self.train_epoch(train_loader=self.parser.get_train_set(),
                                                     model=self.model,
                                                     criterion=self.criterion,
                                                     evaluator=self.evaluator,
                                                     optimizer=self.optimizer,
                                                     epoch=epoch,
                                                     scheduler=self.scheduler,
                                                     report=self.ARCH["train"]["report_batch"],
                                                     show_scans=self.ARCH["train"]["show_scans"])
      # update info
      self.info["train_update"] = update_mean
      self.info["train_loss"] = loss
      if epoch % self.ARCH["train"]["report_epoch"] == 0:
        # evaluate on validation set
        print("*" * 80)
        loss, f1, rand_img  = self.validate(val_loader=self.parser.get_valid_set(),
                                                 model=self.model,
                                                 criterion=self.criterion,
                                                 evaluator=self.evaluator,
                                                 save_scans=self.ARCH["train"]["save_scans"])

        # update info
        self.info["valid_loss"] = loss
        # remember best iou and save checkpoint
        if f1 > best_f1_score:
          print("Best f1 score in validation so far, save model!")
          print("*" * 80)
          best_f1_score = f1

          # save the weights!
          self.model_single.save_checkpoint(self.log, suffix="")

        print("*" * 80)
        # save to log
        Trainer.save_to_log(logdir=self.log,
                            logger=self.tb_logger,
                            info=self.info,
                            epoch=epoch,
                            w_summary=self.ARCH["train"]["save_summary"],
                            model=self.model_single,
                            img_summary=self.ARCH["train"]["save_scans"],
                            imgs=rand_img)

    print('Finished Training')

    return

  def train_epoch(self, train_loader, model, criterion, evaluator, optimizer, epoch, scheduler, report=10, show_scans=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    f1 = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()
    update_ratio_meter = AverageMeter()

    # empty the cache to train now
    if self.gpu:
      torch.cuda.empty_cache()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (in_vol, proj_mask, proj_labels, path_seq, path_name) in enumerate(train_loader):
        # measure data loading time
      data_time.update(time.time() - end)
      if not self.multi_gpu and self.gpu:
        in_vol = [x.cuda() for x in in_vol]
        proj_mask = proj_mask.cuda()
      if self.gpu:
        proj_labels = proj_labels.cuda(non_blocking=True)

      # compute output
      output = model(in_vol, proj_mask)
      loss = criterion(output, proj_labels)

      # compute gradient and do SGD step
      optimizer.zero_grad()
      if self.n_gpus > 1:
        idx = torch.ones(self.n_gpus).cuda()
        loss.backward(idx)
      else:
        loss.backward()
      optimizer.step()

      # measure accuracy and record loss
      loss = loss.mean()
      with torch.no_grad():
        evaluator.reset()
        pred = (output > 0.5).long()
        evaluator.addBatch(pred, proj_labels)
        p, r, f = evaluator.getScores()
      batch_size = in_vol[0].size(0)
      losses.update(loss.item(), batch_size)
      precision.update(p.item(), batch_size)
      recall.update(r.item(), batch_size)
      f1.update(f.item(), batch_size)
      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      # get gradient updates and weights, so I can print the relationship of
      # their norms
      update_ratios = []
      for g in self.optimizer.param_groups:
        lr = g["lr"]
        for value in g["params"]:
          if value.grad is not None:
            w = np.linalg.norm(value.data.cpu().numpy().reshape((-1)))
            update = np.linalg.norm(-max(lr, 1e-10) *
                                    value.grad.cpu().numpy().reshape((-1)))
            update_ratios.append(update / max(w, 1e-10))
      update_ratios = np.array(update_ratios)
      update_mean = update_ratios.mean()
      update_std = update_ratios.std()
      update_ratio_meter.update(update_mean)  # over the epoch

      # if show_scans:
        # get the first scan in batch and project points
        # mask_np = proj_mask[0].cpu().numpy()
        # depth_np = in_vol[0][0].cpu().numpy()
        # pred_np = output[0].cpu().numpy()
        # gt_np = proj_labels[0].cpu().numpy()
        # out = Trainer.make_log_img(depth_np, mask_np, pred_np, gt_np, color_fn)
        # cv2.imshow("sample_training", out)
        # cv2.waitKey(1)

      if i % self.ARCH["train"]["report_batch"] == 0:
        print('Lr: {lr:.3e} | '
              'Update: {umean:.3e} mean,{ustd:.3e} std | '
              'Epoch: [{0}][{1}/{2}] | '
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
              'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
              'precision {precision.val:.3f} ({precision.avg:.3f}) | '
              'recall {recall.val:.3f} ({recall.avg:.3f}) | '
              'f1 score {f1.val:.3f} ({f1.avg:.3f})'.format(
                  epoch, i, len(train_loader), batch_time=batch_time,
                  loss=losses, precision=precision, recall=recall, f1=f1,
                  lr=lr, umean=update_mean, ustd=update_std))

      # step scheduler
      scheduler.step()

    return losses.avg, update_ratio_meter.avg

  def validate(self, val_loader, model, criterion, evaluator, save_scans):
    batch_time = AverageMeter()
    losses = AverageMeter()
    f1 = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()
    rand_imgs = []

    # switch to evaluate mode
    model.eval()

    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      end = time.time()
      for i, (in_vol, proj_mask, proj_labels, path_seq, path_name) in enumerate(val_loader):
        if not self.multi_gpu and self.gpu:
          in_vol = [x.cuda() for x in in_vol]
          proj_mask = proj_mask.cuda()
        if self.gpu:
          proj_labels = proj_labels.cuda(non_blocking=True)

        # compute output
        output = model(in_vol, proj_mask)
        loss = criterion(output, proj_labels)
        pred = (output > 0.5).long()
        # measure accuracy and record loss
        evaluator.addBatch(pred, proj_labels)
        batch_size = in_vol[0].size(0)
        losses.update(loss.mean().item(), batch_size)

        # if save_scans:
          # get the first scan in batch and project points
          # mask_np = proj_mask[0].cpu().numpy()
          # depth_np = in_vol[0][0].cpu().numpy()
          # pred_np = output[0].cpu().numpy()
          # gt_np = proj_labels[0].cpu().numpy()
          # out = Trainer.make_log_img(depth_np,
          #                            mask_np,
          #                            pred_np,
          #                            gt_np,
          #                            color_fn)
          # rand_imgs.append(out)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

      p, r, f1 = evaluator.getScores()
      print('Validation set:\n'
            'Time avg per batch {batch_time.avg:.3f}\n'
            'Loss avg {loss.avg:.4f}\n'
            'Precision {precision:.4f}\n'
            'Recall {recall:.4f}\n'
            'F1 Score {f1:.4f}'.format(batch_time=batch_time,
                                           loss=losses,
                                           precision=p,
                                           recall=r,
                                           f1=f1))
    return losses.avg, f1, rand_imgs
