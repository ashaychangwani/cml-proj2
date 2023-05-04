import models
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import time
import logging
import torch.nn.functional as F
from PIL import Image



async def update_model(data_dir):
    print("starting to update the model")
    model = models.__dict__['multi_resnet50_kd'](num_classes=100)
    model = torch.nn.DataParallel(model).to('cpu')

    checkpoint_path = 'save_checkpoints/multi_resnet50_kd/checkpoint_latest.pth.tar'
    if os.path.isfile(checkpoint_path):
        print("=> loading checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        print('=> loaded checkpoint `{}` (epoch: {})'.format(
            checkpoint_path, checkpoint['epoch']))
    else:
        print('=> no checkpoint found at `{}`'.format(checkpoint_path))
        return

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010]),
    ])
    train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, transform=train_transform, download=True) #The download = True should not be necessary, but in spite of uploading the correct dataset, the code is giving an error
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=128, 
                                                shuffle=True, 
                                                num_workers=4, 
                                                pin_memory=True)

    criterion = nn.CrossEntropyLoss().to('cpu')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay = 5e-4)

    end = time.time()
    model.train()

    for current_epoch in range(0, 5):
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

        adjust_learning_rate(optimizer, current_epoch)

        for i, (input, target) in enumerate(train_loader):
            data_time.update(time.time() - end)
            
            target = target.squeeze().long().to('cpu')
            input = input.to('cpu')

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

            temp4 = output / 3
            temp4 = torch.softmax(temp4, dim=1)
            
            
            loss1by4 = kd_loss_function(middle_output1, temp4.detach()) * (3**2)
            losses1_kd.update(loss1by4, input.size(0))
            
            loss2by4 = kd_loss_function(middle_output2, temp4.detach()) * (3**2)
            losses2_kd.update(loss2by4, input.size(0))
            
            loss3by4 = kd_loss_function(middle_output3, temp4.detach()) * (3**2)
            losses3_kd.update(loss3by4, input.size(0))
            
            feature_loss_1 = feature_loss_function(middle1_fea, final_fea.detach()) 
            feature_losses_1.update(feature_loss_1, input.size(0))
            feature_loss_2 = feature_loss_function(middle2_fea, final_fea.detach()) 
            feature_losses_2.update(feature_loss_2, input.size(0))
            feature_loss_3 = feature_loss_function(middle3_fea, final_fea.detach()) 
            feature_losses_3.update(feature_loss_3, input.size(0))

            total_loss = (1 - 0.1) * (loss + middle1_loss + middle2_loss + middle3_loss) + \
                        0.1 * (loss1by4 + loss2by4 + loss3by4) + \
                        1e-6 * (feature_loss_1 + feature_loss_2 + feature_loss_3)
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

                      
            print("Epoch: [{0}]\t"
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

        checkpoint_path = os.path.join('save_checkpoints/multi_resnet50_kd/', 'checkpoint_{:05d}.pth.tar'.format(current_epoch))
        save_checkpoint({
            'epoch': current_epoch,
            'arch': 'multi_resnet50_kd',
            'state_dict': model.state_dict(),
        }, filename=checkpoint_path)
        shutil.copyfile(checkpoint_path, os.path.join('save_checkpoints/multi_resnet50_kd/', 'checkpoint_latest.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    if (epoch < 1):
        lr = 0.01
    elif 75 <= epoch < 130:
        lr = 0.1 * (0.1 ** 1)
    elif 130 <= epoch < 180:
        lr = 0.1 * (0.1 ** 2)
    elif epoch >=180:
        lr = 0.1 * (0.1 ** 3)
    else:
        lr = 0.1

    print('Epoch [{}] learning rate = {}'.format(epoch, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def kd_loss_function(output, target_output):
    """Compute kd loss"""
    """
    para: output: middle ouptput logits.
    para: target_output: final output has divided by temperature and softmax.
    """

    output = output / 3
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
    return loss_kd

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

def feature_loss_function(fea, target_fea):
    loss = (fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()

def predict_class(image, device='cpu'):
    # Load image and apply transformations
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0).to(device)
    model = models.__dict__['multi_resnet50_kd'](num_classes=100)
    model = torch.nn.DataParallel(model).to('cpu')
    checkpoint_path = 'save_checkpoints/multi_resnet50_kd/checkpoint_latest.pth.tar'
    
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    # Make prediction
    model.eval()
    with torch.no_grad():
        output, _, _, _, _, _, _, _ = model(image)
        _, predicted_class = torch.max(output.data, 1)
    
    return predicted_class.item()