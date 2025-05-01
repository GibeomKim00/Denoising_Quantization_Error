import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_data_loaders(dataset='cifar10', batch_size=64, augment=False):
    if dataset == 'cifar10':
        normalize_mean = (0.4914, 0.4822, 0.4465)
        normalize_std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'imagenet':
        normalize_mean = (0.485, 0.456, 0.406)
        normalize_std = (0.229, 0.224, 0.225)
    elif dataset == "cifar100":
        normalize_mean = (0.5071, 0.4865, 0.4409)
        normalize_std = (0.2673, 0.2564, 0.2762)
    else:
        raise ValueError("Unsupported dataset. Choose 'cifar10' or 'imagenet' or 'cifar100'.")

    # 기본 변환 설정
    base_transform = [
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ]

    # 데이터 증강이 필요할 경우 추가 (augment=True 시)
    if augment:
        if dataset in ["cifar10", "cifar100"]:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                *base_transform
            ])
        elif dataset == "imagenet":
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                *base_transform
            ])
    else:
        train_transform = transforms.Compose(base_transform)
    
    # 테스트용 변환 (증강 없이 정규화만 적용)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])

    # 데이터셋 로드
    if dataset == 'cifar10':
        data_root = "/local_dataset"
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=test_transform)
    elif dataset == 'imagenet':
        data_root = "/local_dataset/ImageNet"
        train_dataset = torchvision.datasets.ImageNet(
            root=data_root, split='train', transform=train_transform)
        test_dataset = torchvision.datasets.ImageNet(
            root=data_root, split='val', transform=test_transform)
    elif dataset == "cifar100":
        data_root = "/local_dataset"
        train_dataset = torchvision.datasets.CIFAR100(
            root=data_root, train=True, download=True, transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR100(
            root=data_root, train=False, download=True, transform=test_transform)

    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader