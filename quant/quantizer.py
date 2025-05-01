import torch
from torch import nn
from torch.nn import functional as F

def uniform_affine_quantizer_weight(weights, wq=4):
    n_levels = 2 ** wq
    w_min = torch.min(weights)
    w_max = torch.max(weights)

    if w_max == w_min:
        return weights
    
    scale = (w_max - w_min) / (n_levels - 1)
    zero_point = torch.round(-w_min / scale)

    quantized_weights = torch.round(weights / scale + zero_point)
    quantized_weights = torch.clamp(quantized_weights, 0, n_levels - 1)

    dequantized_weights = scale * (quantized_weights - zero_point)

    return dequantized_weights

def uniform_affine_quantizer_activation(activations, aq=4):
    n_levels = 2 ** aq
    a_min = torch.min(activations)
    a_max = torch.max(activations)

    if a_max == a_min:
        return activations
    
    scale = (a_max - a_min) / (n_levels - 1)
    zero_point = torch.round(-a_min / scale)

    quantized_activations = torch.round(activations / scale + zero_point)
    quantized_activations = torch.clamp(quantized_activations, 0, n_levels - 1)

    dequantized_activations = scale * (quantized_activations - zero_point)

    return dequantized_activations