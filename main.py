import argparse
import datetime
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import copy

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

from models.student import get_student_model
from models.resnet import get_resnet_model
from utils.training import train, test, train_denoising
from data.dataloader import get_data_loaders
from config import Config
from quant.quant_model import QuantModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"Checkpoint saved at {filename}")


def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    if os.path.isfile(filename):
        print(f"Loading checkpoint from {filename}")
        checkpoint = torch.load(filename, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"Loaded checkpoint from epoch {epoch} with best_acc {best_acc:.3f}")
        return epoch, best_acc
    else:
        print(f"No checkpoint found at {filename}")
        return 0, 0.0


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, choices=[18, 34, 50, 101, 110, 152], required=True,
                        help='ResNet depth (18, 34, 50, 101, 110, 152)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'imagenet'], default='cifar10',
                        help='Dataset to use (cifar10 or imagenet)')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--l', type=str, default='./exp_results')

    parser.add_argument("--weight_dir", action="store_true")
    parser.add_argument("--ptq", action="store_true")
    parser.add_argument("--wq", type=int, default=4)
    parser.add_argument("--aq", type=int, default=4)
    parser.add_argument("--denoise_layer1", action="store_true")
    parser.add_argument("--denoise_layer2", action="store_true")
    return parser

def main():
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)
    result_str = ""

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    model_name = f"resnet{opt.n}"
    log_dir = os.path.join(opt.l, model_name, opt.dataset)
    weight_dir = os.path.join(opt.save_dir, opt.dataset)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)

    config = Config(model_name, opt.dataset)
    
    # CUDA 및 시드 설정
    cuda = torch.cuda.is_available()
    cudnn.benchmark = True
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
    
    # 데이터 로더
    train_loader, test_loader = get_data_loaders(
        dataset=opt.dataset,
        batch_size=config.batch_size,
        augment=config.data_augmentation
    )

    # Teacher model
    teacher_model = get_resnet_model(
        model_type=str(opt.n),
        num_classes=10 if opt.dataset == 'cifar10' else 1000, 
        cifar10 = opt.dataset == 'cifar10'
    )
    if cuda:
        teacher_model = teacher_model.cuda()

    # 학습된 가중치가 없을 경우
    state_dict = None
    if opt.weight_dir is False:    
        optimizer = optim.SGD(
            teacher_model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        if cuda:
            criterion = criterion.cuda()
        scheduler = MultiStepLR(optimizer, milestones=config.lr_milestones, gamma=0.1)

        checkpoint_path = os.path.join(weight_dir, f'{model_name}_checkpoint.pth')
        start_epoch, best_train_acc = load_checkpoint(checkpoint_path, teacher_model, optimizer, scheduler)
        
        # training
        for epoch in range(start_epoch, config.epochs):
            print(f"\nEpoch {epoch+1}/{config.epochs}")
            train_acc = train(train_loader=train_loader,
                            epoch=epoch,
                            model=teacher_model,
                            optimizer=optimizer,
                            criterion=criterion,
                            print_freq=config.print_freq,
                        )
            
            scheduler.step()
            print(f"Current Learning Rate: {scheduler.get_last_lr()[0]}")  

            if (epoch + 1) % 20 == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': teacher_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': max(best_train_acc, train_acc)
                }, checkpoint_path)

            if train_acc > best_train_acc:
                best_train_acc = train_acc
                torch.save(teacher_model.state_dict(), os.path.join(weight_dir, f'{model_name}.pth'))
                state_dict = teacher_model.state_dict()
                print(f'Best model saved with Train Acc@1: {best_train_acc:.3f}')

        print('\n=== Training Completed, Starting Final Test ===')
    else:
        state_dict = torch.load(os.path.join(weight_dir, f'{model_name}.pth'), weights_only=False)
        teacher_model.load_state_dict(state_dict)

    teacher_test_acc1, teacher_test_acc5 = test(
        test_loader=test_loader,
        epoch=config.epochs,
        model=teacher_model
    )
    
    teacher_param_count = count_parameters(teacher_model)
    result_str += f"[Teacher Model] #Params: {teacher_param_count:,}\n"

    result_str += f"[Teacher Model] Acc@1: {teacher_test_acc1:.3f} || Acc@5: {teacher_test_acc5:.3f}\n"


    # 양자화를 할 경우
    if opt.ptq:
        student_model = get_student_model(
                            model_type=str(opt.n),
                            num_classes=10 if opt.dataset == 'cifar10' else 1000, 
                            cifar10 = opt.dataset == 'cifar10'
                        )
        if cuda:
            student_model = student_model.cuda()
        student_model.load_state_dict(state_dict, strict=False)

        student_model = QuantModel(
            model=student_model,
            wq = opt.wq,
            aq = opt.aq,
        )

        teacher_quant = QuantModel(
            model = teacher_model,
            wa = opt.wq,
            aq = opt.aq
        )

        teacher_quant_test_acc1, teacher_quant_test_acc5 = test(
            test_loader=test_loader,
            epoch=config.epochs,
            model=teacher_quant
        )
        result_str += f"[Student Quantized({opt.wq}_{opt.aq})] Acc@1: {teacher_quant_test_acc1:.3f} || Acc@5: {teacher_quant_test_acc5:.3f}\n"

        
        for p in student_model.parameters():
            p.requires_grad = False
        for p in student_model.model.denoising_layer1.parameters():
            p.requires_grad = True
        
        optimizer = optim.SGD(
                        student_model.model.denoising_layer1.parameters(),
                        lr=config.learning_rate,
                        momentum=config.momentum,
                        weight_decay=config.weight_decay
                    )
        
        criterion = nn.MSELoss()
        scheduler = MultiStepLR(optimizer, milestones=config.lr_milestones, gamma=0.1)
           
        best_loss_denoising = float('inf')
        for epoch in range(config.epochs):
            print(f"\nEpoch {epoch+1}/{config.epochs}")
            train_loss_denoising = train_denoising(
                                    train_loader=train_loader,
                                    epoch=epoch,
                                    model=student_model,
                                    optimizer=optimizer,
                                    mode='denoise1',
                                    criterion=criterion,
                                    teacher_model=teacher_model,
                                    print_freq=config.print_freq
                                )
            scheduler.step()
            print(f"Current Learning Rate: {scheduler.get_last_lr()[0]}")                 

            if train_loss_denoising < best_loss_denoising:
                best_loss_denoising = train_loss_denoising
                torch.save(student_model.state_dict(), os.path.join(weight_dir, f'student_{model_name}_denoise.pth'))
                print(f'Best model saved with lowest MSE Loss: {best_loss_denoising:.6f}')
    
        '''
        for p in student_model.parameters():
            p.requires_grad = False

        for p in student_model.model.denoising_layer2.parameters():
            p.requires_grad = True

        optimizer = optim.SGD(
                        student_model.model.denoising_layer2.parameters(),
                        lr=config.learning_rate,
                        momentum=config.momentum,
                        weight_decay=config.weight_decay
                    )
        criterion = nn.CrossEntropyLoss()
        scheduler = MultiStepLR(optimizer, milestones=config.lr_milestones, gamma=0.1)

        best_acc_denoising2 = 0.0
        print("Training denoising_layer2...\n")

        for epoch in range(config.epochs):
            print(f"\nEpoch {epoch+1}/{config.epochs}")

            train_acc_denoising2 = train_denoising(
                train_loader=train_loader,
                epoch=epoch,
                model=student_model,
                optimizer=optimizer,
                mode='denoise2',
                criterion=criterion,
                teacher_model=None,
                print_freq=config.print_freq
            )

            scheduler.step()
            print(f"Current Learning Rate: {scheduler.get_last_lr()[0]}")   

            if train_acc_denoising2 > best_acc_denoising2:
                best_acc_denoising2 = train_acc_denoising2
                torch.save(student_model.state_dict(), os.path.join(weight_dir, f'student_{model_name}_denoise2.pth'))
                print(f'✅ Best model saved with Train Acc@1: {best_acc_denoising2:.3f}')
        '''

        student_denoise_test_acc1, student_denoise_test_acc5 = test(
            test_loader=test_loader,
            epoch=config.epochs,
            model=student_model
        )
       
        # student_param_count = count_parameters(student_model)
        # result_str += f"[Student Model] #Params: {student_param_count:,}\n"
        
        result_str += f"[Student Quantized + denoising] Acc@1: {student_denoise_test_acc1:.3f} || Acc@5: {student_denoise_test_acc5:.3f}\n"


    # 결과값을 log_dir에 저장
    result_file = os.path.join(log_dir, f'result_denoise1_only.txt')
    with open(result_file, 'w') as f:
        f.write(result_str)


if __name__ == "__main__":
    main()