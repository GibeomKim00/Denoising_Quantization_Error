import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

from quant.quantizer import uniform_affine_quantizer_weight, uniform_affine_quantizer_activation

EXCLUDE_FROM_QUANT = ["denoising_layer1", "denoising_layer2"]


class QuantConv2d(nn.Module):
    def __init__(self, conv_layer: nn.Conv2d, wq: int, aq: int):
        super().__init__()
        self.wq = wq
        self.aq = aq
        self.conv = conv_layer

    def forward(self, x):
        quantized_weight = uniform_affine_quantizer_weight(self.conv.weight, self.wq)
        output = F.conv2d(
            x,
            quantized_weight,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups
        )

        output_quant = uniform_affine_quantizer_activation(output, self.aq)
        return output_quant
    
class QuantLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, wq: int, aq: int):
        super().__init__()
        self.linear = linear_layer
        self.wq = wq
        self.aq = aq

    def forward(self, x):
        quantized_weight = uniform_affine_quantizer_weight(self.linear.weight, self.wq)
        output =  F.linear(x, quantized_weight, self.linear.bias)

        output_quant = uniform_affine_quantizer_activation(output, self.aq)
        return output_quant

class QuantModel(nn.Module):
    
    def __init__(
        self,
        model: nn.Module,
        wq: int = 4,
        aq: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        self.wq = wq
        self.aq = aq

        self.model = self._wrap_model(model)

    def _wrap_model(self, model: nn.Module) -> nn.Module:
        for name, module in model.named_children():
            if name in EXCLUDE_FROM_QUANT:
                continue

            if isinstance(module, nn.Conv2d):
                setattr(model, name, QuantConv2d(module, self.wq, self.aq))
            elif isinstance(module, nn.Linear):
                setattr(model,name, QuantLinear(module, self.wq, self.aq))
            else:
                self._wrap_model(module)
        return model
    
    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)
