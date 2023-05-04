import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import os
import argparse
import time
import logging
import models.resnet as models
from dataset.data import *
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 training')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'])
    parser.add_argument('--data-dir', default='dataset', type=str,
                        help='the diretory to save cifar100 dataset')
    parser.add_argument('--arch', metavar='ARCH', default='multi_resnet50_kd',
                        help='model architecture')
    parser.add_argument('--dataset', '-d', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100'],
                        help='dataset choice')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4 )')
    parser.add_argument('--epoch', default=200, type=int,
                        help='number of total iterations (default: 64,000)')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual iter number (useful on restarts)')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=5e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', default=100, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to  latest checkpoint (default: None)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pretrained model')
    parser.add_argument('--step-ratio', default=0.1, type=float,
                        help='ratio for learning rate deduction')
    parser.add_argument('--warm-up', action='store_true',
                        help='for n = 18, the model needs to warm up for 400 '
                             'iterations')
    parser.add_argument('--save-folder', default='save_checkpoints/', type=str,
                        help='folder to save the checkpoints')

    #kd parameter
    parser.add_argument('--temperature', default=3, type=int,
                        help='temperature to smooth the logits')
    parser.add_argument('--alpha', default=0.1, type=float,
                        help='weight of kd loss')
    parser.add_argument('--beta', default=1e-6, type=float,
                        help='weight of feature loss')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    save_path = args.save_path = os.path.join(args.save_folder, args.arch)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    args.logger_file = os.path.join(save_path, 'log_training.txt')
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    logging.info('start training {}'.format(args.arch))
    run_training(args)

def run_training(args):
    if args.dataset == 'cifar100':
        model = models.__dict__[args.arch](num_classes=100)
    else:
        raise NotImplementedError
    model = torch.nn.DataParallel(model).to(args.device)

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint `{}`".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (epoch: {})'.format(
                args.resume, checkpoint['epoch']))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = True
    if args.dataset == 'cifar100':
        train_loader = prepare_cifar100_train_dataset(data_dir=args.data_dir, batch_size=args.batch_size, 
                                                        num_workers=args.workers)
    else:
        raise NotImplementedError

    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.weight_decay)

    end = time.time()
    model.train()

    for current_epoch in range(args.start_epoch, args.epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        middle1_losses = AverageMeter()
        middle2_losses = AverageMeter()
        middle3_losses = AverageMeter()
        losses1_kd = AverageMeter()
        losses2_kd = AverageMeter()
        losses3_kd = AverageMeter()
        feature_losses_1 = AverageMeter()
        feature_losses_2 = AverageMeter()
        feature_losses_3 = AverageMeter()
        total_losses = AverageMeter()
        middle1_top1 = AverageMeter()
        middle2_top1 = AverageMeter()
        middle3_top1 = AverageMeter()

        adjust_learning_rate(args, optimizer, current_epoch)

        for i, (input, target) in enumerate(train_loader):
            data_time.update(time.time() - end)
            
            target = target.squeeze().long().to(args.device)
            input = input.to(args.device)

            output, middle_output1, middle_output2, middle_output3, \
            final_fea, middle1_fea, middle2_fea, middle3_fea = model(input)
            
            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))

            middle1_loss = criterion(middle_output1, target)
            middle1_losses.update(middle1_loss.item(), input.size(0))
            middle2_loss = criterion(middle_output2, target)
            middle2_losses.update(middle2_loss.item(), input.size(0))
            middle3_loss = criterion(middle_output3, target)
            middle3_losses.update(middle3_loss.item(), input.size(0))

            temp4 = output / args.temperature
            temp4 = torch.softmax(temp4, dim=1)
            
            
            loss1by4 = kd_loss_function(middle_output1, temp4.detach(), args) * (args.temperature**2)
            losses1_kd.update(loss1by4, input.size(0))
            
            loss2by4 = kd_loss_function(middle_output2, temp4.detach(), args) * (args.temperature**2)
            losses2_kd.update(loss2by4, input.size(0))
            
            loss3by4 = kd_loss_function(middle_output3, temp4.detach(), args) * (args.temperature**2)
            losses3_kd.update(loss3by4, input.size(0))
            
            feature_loss_1 = feature_loss_function(middle1_fea, final_fea.detach()) 
            feature_losses_1.update(feature_loss_1, input.size(0))
            feature_loss_2 = feature_loss_function(middle2_fea, final_fea.detach()) 
            feature_losses_2.update(feature_loss_2, input.size(0))
            feature_loss_3 = feature_loss_function(middle3_fea, final_fea.detach()) 
            feature_losses_3.update(feature_loss_3, input.size(0))

            total_loss = (1 - args.alpha) * (loss + middle1_loss + middle2_loss + middle3_loss) + \
                        args.alpha * (loss1by4 + loss2by4 + loss3by4) + \
                        args.beta * (feature_loss_1 + feature_loss_2 + feature_loss_3)
            total_losses.update(total_loss.item(), input.size(0))
            
            prec1 = accuracy(output.data, target, topk=(1,))
            top1.update(prec1[0], input.size(0))

            middle1_prec1 = accuracy(middle_output1.data, target, topk=(1,))
            middle1_top1.update(middle1_prec1[0], input.size(0))
            middle2_prec1 = accuracy(middle_output2.data, target, topk=(1,))
            middle2_top1.update(middle2_prec1[0], input.size(0))
            middle3_prec1 = accuracy(middle_output3.data, target, topk=(1,))
            middle3_top1.update(middle3_prec1[0], input.size(0))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:             
                logging.info("Epoch: [{0}]\t"
                            "Iter: [{1} / {2}]\t"
                            "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                            "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                            "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                            "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t".format(
                                current_epoch,
                                i,
                                len(train_loader),
                                batch_time=batch_time,
                                data_time=data_time,
                                loss=total_losses,
                                top1=top1)
                ) 
        print("Out of the loop")
        checkpoint_path = os.path.join(args.save_path, 'checkpoint_{:05d}.pth.tar'.format(current_epoch))
        save_checkpoint({
            'epoch': current_epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
        }, filename=checkpoint_path)
        shutil.copyfile(checkpoint_path, os.path.join(args.save_path, 'checkpoint_latest.pth.tar'))
        torch.cuda.empty_cache()

def adjust_learning_rate(args, optimizer, epoch):
    if args.warm_up and (epoch < 1):
        lr = 0.01
    elif 75 <= epoch < 130:
        lr = args.lr * (args.step_ratio ** 1)
    elif 130 <= epoch < 180:
        lr = args.lr * (args.step_ratio ** 2)
    elif epoch >=180:
        lr = args.lr * (args.step_ratio ** 3)
    else:
        lr = args.lr

    logging.info('Epoch [{}] learning rate = {}'.format(epoch, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)  
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))  

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul(100.0 / batch_size))

    return res

def save_checkpoint(state, filename):
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

def kd_loss_function(output, target_output,args):
    """Compute kd loss"""
    """
    para: output: middle ouptput logits.
    para: target_output: final output has divided by temperature and softmax.
    """

    output = output / args.temperature
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
    return loss_kd
def feature_loss_function(fea, target_fea):
    loss = (fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()


if __name__ == '__main__':
    main()