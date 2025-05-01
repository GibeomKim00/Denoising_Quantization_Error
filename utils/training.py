import torch
import torch.nn as nn
import time
from torch.nn import functional as F

from .metrics import AverageMeter, ProgressMeter, accuracy


def train(train_loader, epoch, model, optimizer, criterion, print_freq=100):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, losses, top1, top5, prefix=f"Epoch: [{epoch}]")
    
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input, target = input.cuda(), target.cuda()
        output = model(input)
        loss = criterion(output, target)
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0].item(), input.size(0))
        top5.update(acc5[0].item(), input.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % print_freq == 0:
            progress.print(i)
    
    print(f'==> Train Accuracy: Acc@1 {top1.avg:.3f} || Acc@5 {top5.avg:.3f}')
    return top1.avg

def test(test_loader, epoch, model):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input, target = input.cuda(), target.cuda()
            output = model(input)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0].item(), input.size(0))
            top5.update(acc5[0].item(), input.size(0))
    print(f'==> Test Accuracy: Acc@1 {top1.avg:.3f} || Acc@5 {top5.avg:.3f}')
    return top1.avg, top5.avg 


def train_denoising(train_loader, epoch, model, optimizer, mode, criterion, teacher_model=None, print_freq=100):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, losses, top1, top5, prefix=f"Epoch: [{epoch}]")

    model.train()
    if teacher_model is not None:
        teacher_model.eval()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input, target = input.cuda(), target.cuda()

        if mode == 'denoise1':
            with torch.no_grad():
                teacher_feat = teacher_model(input, return_feature=True)
            student_feat = model(input, mode='denoise1')
            loss = criterion(student_feat, teacher_feat)

            acc1, acc5 = torch.tensor(0.0), torch.tensor(0.0)  # 정확도 의미 없음

        elif mode == 'denoise2':
            output = model(input, mode='denoise2')
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.print(i)

    print(f'==> Train ({mode}) Result: Loss {losses.avg:.4e} || Acc@1 {top1.avg:.3f} || Acc@5 {top5.avg:.3f}')

    if mode == 'denoise1':
        return losses.avg  
    else:
        return top1.avg    