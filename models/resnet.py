import torch
import torch.nn as nn
from .blocks import BasicBlock, Bottleneck


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, cifar10=False, denoising=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64 if not cifar10 else 16  # CIFAR-10용은 16으로 시작
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple")
        self.groups = groups
        self.base_width = width_per_group
        self.cifar10 = cifar10

        # CIFAR-10용 초기 레이어는 3x3 conv, ImageNet용은 7x7 conv + maxpool
        if cifar10:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # 레이어 구성
        if cifar10:
            self.layer1 = self._make_layer(block, 16, layers[0])
            self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64 * block.expansion, num_classes)
        else:
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)


        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x, return_feature=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.cifar10:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if not self.cifar10:
            x = self.layer4(x)

        if return_feature:
            return x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def get_resnet_model(model_type, num_classes=10, cifar10=False):
    # CIFAR-10과 ImageNet을 위한 모델 매핑
    model_map = {
        # ImageNet용 모델 (cifar10=False)
        '18': lambda cifar10: ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, cifar10=cifar10),
        '34': lambda cifar10: ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, cifar10=cifar10),
        '50': lambda cifar10: ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, cifar10=cifar10),
        '101': lambda cifar10: ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, cifar10=cifar10),
        '152': lambda cifar10: ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, cifar10=cifar10),
        # CIFAR-10용 모델 (cifar10=True)
        '110': lambda cifar10: ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes, cifar10=True) if cifar10 else None
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unsupported model type: {model_type}. Choose from {list(model_map.keys())}")
    
    if model_type == '110' and not cifar10:
        raise ValueError("ResNet-110 is only supported for CIFAR-10 (set cifar10=True)")
    
    return model_map[model_type](cifar10)

# 사용 예시:
# ImageNet용 ResNet-50
# model = get_resnet_model('50', num_classes=1000, cifar10=False)
# CIFAR-10용 ResNet-110
# model = get_resnet_model('110', num_classes=10, cifar10=True)