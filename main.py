
import os
import argparse
import time
import yaml
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from FeatherNet.utils.profile import count_params
from FeatherNet.utils.data_aug import ColorAugmentation
from torch.autograd.variable import Variable
# sklearn libs
from sklearn.metrics import confusion_matrix
from FeatherNet.tools import roc
import config
from FeatherNet import models
from FeatherNet.tools.dataset import FaceQualityDataset
from FeatherNet.tools.losses import *
from FeatherNet.tools.benchmark import compute_speed, stat
import cv2
from PIL import Image


def write_log(log_file, mess):
    if os.path.exists(log_file):
        write_log = open(log_file, 'a', encoding='utf-8')
    else:
        write_log = open(log_file, 'w', encoding='utf-8')
    write_log.write(mess)
    write_log.close()


def main():

    global best_prec1, USE_GPU, device
    best_prec1 = 0
    USE_GPU = torch.cuda.is_available()
    ## Set random seeds ##
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    model = models.__dict__[config.arch]()

    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else "cpu")

    str_input_size = '1x3x224x224'
    if config.summary:
        input_size = tuple(int(x) for x in str_input_size.split('x'))
        stat(model,input_size)
        return
    if USE_GPU:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(config.random_seed)
        # args.gpus = [int(i) for i in args.gpus.split(',')]
        # model = torch.nn.DataParallel(model)
        model.to(device)

    # count_params(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('total_params', pytorch_total_params)

    # define loss function (criterion) and optimizer
    criterion = FocalLoss(device, config.class_num, gamma=config.fl_gamma)

    optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)
    
    if config.speed:
        input_size = tuple(int(x) for x in str_input_size.split('x') )
        iteration = config.test_iteration
        compute_speed(model, input_size, device, iteration)
        return

    # optionally resume from a checkpoint
    if config.resume:
        print(os.getcwd())
        if os.path.isfile(config.resume):
            print("=> loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume)
            config.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(config.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(config.resume))


    # Data loading code
    normalize = transforms.Normalize(mean=config.mean,  ##accorcoding to casia-surf val to commpute
                                     std=config.std)
    img_size = config.input_size

    train_dataset = FaceQualityDataset(
        root=config.root,
        ann_file=config.train_file,
        transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ColorAugmentation(),
            normalize,
        ]))
    val_dataset = FaceQualityDataset(
        root=config.root,
        ann_file=config.val_file,
        transform=transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
    # test_dataset = FaceQualityDataset(
    #     root=config.root,
    #     ann_file=config.test_file,
    #     transform=transforms.Compose([
    #     transforms.Resize((img_size, img_size)),
    #     transforms.ToTensor(),
    #     normalize,
    # ]))
    train_sampler = None
    val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.workers, pin_memory=(train_sampler is None), sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True,
                                             num_workers=config.workers, pin_memory=False, sampler=val_sampler)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True,
    #                                          num_workers=config.workers, pin_memory=False, sampler=val_sampler)

    # if config.evaluate:
    #     validate(test_loader, model, criterion, config.start_epoch)
    #     return
    # else:
    #     print(model)

    for epoch in range(0, config.epochs):
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion,epoch)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        if is_best:
            print('epoch: {} The best is {} last best is {}'.format(epoch,prec1,best_prec1))
        best_prec1 = max(prec1, best_prec1)

        m_path = config.checkpoints
        
        if not os.path.exists(m_path):
            os.makedirs(m_path)
        save_name = '{}/{}_{}_best.pth.tar'.format(m_path, config.model_name, epoch) if is_best else\
            '{}/{}_{}.pth.tar'.format(m_path, config.model_name, epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': m_path,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, filename=save_name)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input_var = Variable(input).float().to(device)
        target_var = Variable(target).long().to(device)

        # compute output
        output = model(input_var)
        # print(output)
        loss = criterion(output, target_var)
        prec1, prec2 = accuracy(output.data, target_var,topk=(1,2))

        # measure accuracy and record loss
        reduced_prec1 = prec1.clone()

        top1.update(reduced_prec1[0])

        reduced_loss = loss.data.clone()
        losses.update(reduced_loss)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        #  check whether the network is well connected
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        lr = optimizer.param_groups[0]['lr']

        if i % config.print_freq == 0:
            line = 'Epoch: [{0}][{1}/{2}]\t lr:{3:.5f}\t' \
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                    .format(epoch, i, len(train_loader),lr,
                    batch_time=batch_time, loss=losses, top1=top1)
            print(line)
            write_log(config.log_file, '{}\n'.format(line))


def validate(val_loader, model, criterion,epoch):
    global time_stp
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    result_list = []
    label_list = []
    predicted_list = []

    # switch to evaluate mode
    model.eval()

    end = time.time()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            with torch.no_grad():
                input_var = Variable(input).float().to(device)
                target_var = Variable(target).long().to(device)

                # compute output
                output = model(input_var)
                loss = criterion(output, target_var)

                # measure accuracy and record loss
                prec1, prec2 = accuracy(output.data, target_var,topk=(1,2))
                losses.update(loss.data, input.size(0))
                top1.update(prec1[0], input.size(0))

                soft_output = torch.softmax(output,dim=-1)
                preds = soft_output.to('cpu').detach().numpy()
                label = target.to('cpu').detach().numpy()
                _,predicted = torch.max(soft_output.data, 1)
                predicted = predicted.to('cpu').detach().numpy()

                for i_batch in range(preds.shape[0]):
                    # result_list.append(preds[predicted[i_batch]])
                    label_list.append(label[i_batch])
                    predicted_list.append(predicted[i_batch])

                    # if label[i_batch] != predicted[i_batch]:
                    #     print(path[i_batch]+' '+str(label[i_batch])+' '+ str(predicted[i_batch]))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if i % config.print_freq == 0:
                    line = 'Test: [{0}/{1}]\t' \
                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                           'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                    loss=losses, top1=top1)
                    print(line)

    result_line = 'epoch: {} Acc:{:.3f} '.format(epoch, top1.avg)
    write_log(config.log_file, '{}\n'.format(result_line))
    print(result_line)

    return top1.avg


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = config.lr * (0.1 ** (epoch // config.every_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
