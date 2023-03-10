import os
import math
import datetime
import shutil
import numpy as np
import pytz
import skimage.io
import tqdm 
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import utils


def cross_entropy2d(input, target, weight=None, size_average=True):
    """
    :param: input:(n, c, h, w), target:(n, h, w) 
    """
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    # flatten and select >= 0
    # log_p:(n, c, h, w) -> (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target:(n, h, w) -> (n*h*w, )
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(input=log_p, target=target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss

class Trainer(object):

    def __init__(self, cuda, model, optimizer, 
                 train_dl, val_dl, out, max_iter, 
                 size_average=False, interval_validate=None):
        self.cuda = cuda
        self.model = model
        self.optim = optimizer
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.timestep_start = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
        self.size_average = size_average
        if interval_validate is None:
            self.interval_validate = len(self.train_dl)
        else:
            self.interval_validate = interval_validate
        self.out = out

        if not os.path.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
            'elapsed_time',
        ]
        if not os.path.exists(os.path.join(self.out, 'log.csv')):
            with open(os.path.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_iu = 0

    def validate(self):
        training = self.model.training
        self.model.eval()

        n_class = len(self.val_dl.dataset.class_names)

        val_loss = 0
        visualizations = []
        label_trues, label_preds = [], []
        for batch_idx, (data, target) in tqdm.tqdm(
            enumerate(self.val_dl), total=len(self.val_dl),
            desc='Valid iteration=%d' % self.iteration, ncols=80,
            leave=False
        ):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            with torch.no_grad():
                output = self.model(data)
            loss = cross_entropy2d(output, target, size_average=self.size_average)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while validating')
            val_loss += loss_data / len(data)
        
            images = data.data.cpu()
            label_pred = output.data.max(1)[1].cpu().numpy()[:, :, :]
            label_true = target.data.cpu()
            for img, lt, lp in zip(images, label_true, label_pred):
                img, lt = self.val_dl.dataset.untransform(img, lt)
                label_trues.append(lt)
                label_preds.append(lp)
                if len(visualizations) < 9:
                    viz = utils.visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class
                    )
                    visualizations.append(viz)

        metrics = utils.label_accuracy_score(label_trues, label_preds, n_class)

        out = os.path.join(self.out, 'visualization_viz')
        if not os.path.exists(out):
            os.mkdir(out)
        out_file = os.path.join(out, 'iter%012d.jpg' % self.iteration)
        skimage.io.imsave(out_file, utils.get_tile_image(visualizations))

        val_loss /= len(self.val_dl)

        with open(os.path.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now(pytz.timezone('Asia/Shanghai')) - self.timestep_start
            ).total_seconds()
            log = [self.epoch, self.iteration] + [''] * 5 + \
                  [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, os.path.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(os.path.join(self.out, 'checkpoint.pth.tar'),
                        os.path.join(self.out, 'model_best.pth.tar'))

        if training:
            self.model.train()

    def train_epoch(self):
        self.model.train()

        n_class = len(self.train_dl.dataset.class_names)

        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.train_dl), total=len(self.train_dl),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_dl)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if self.iteration % self.interval_validate == 0:
                self.validate()

            assert self.model.training

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optim.zero_grad()
            score = self.model(data)

            loss = cross_entropy2d(score, target,
                                   size_average=self.size_average)
            loss /= len(data)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optim.step()

            metrics = []
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            acc, acc_cls, mean_iu, fwavacc = \
                utils.label_accuracy_score(
                    lbl_true, lbl_pred, n_class=n_class)
            metrics.append((acc, acc_cls, mean_iu, fwavacc))
            metrics = np.mean(metrics, axis=0)

            with open(os.path.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now(pytz.timezone('Asia/Shanghai')) -self.timestep_start
                ).total_seconds()
                log = [self.epoch, self.iteration] + [loss_data] + \
                    list(metrics) + [''] * 5 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

            if self.iteration >= self.max_iter:
                break

    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_dl)))
        for epoch in tqdm.trange(self.epoch, max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
        print("Finish Training")

            
    
