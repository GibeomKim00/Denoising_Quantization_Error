import os

class Config:
    def __init__(self, model_name, dataset):
        # 모델 이름 설정
        self.model_name = model_name
        
        # 공통 설정
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.seed = 42
        self.data_augmentation = True
        self.print_freq = 100

        # 데이터셋에 따라 다른 설정
        if dataset == 'cifar10':
            self.batch_size = 128
            self.epochs = 200
            self.learning_rate = 0.1
            self.lr_milestones = [100, 150]
        elif dataset == 'imagenet':
            self.batch_size = 128
            self.epochs = 90
            self.learning_rate = 0.1
            self.lr_milestones = [30, 60]
        elif dataset == "cifar100":
            self.batch_size = 128
            self.epochs = 200
            self.learning_rate = 0.1
            self.lr_milestones = [60, 120, 160]
            self.weight_decay = 5e-4
