from inspect import isfunction
import torch, torch.nn as nn
import torch.nn.functional as F

def get_activation(activation, inplace=True):
    """
    Create activation layer from string/function.

    Args
    ----------
    activation (function, or str, or nn.Module):
        Activation function or name of activation function.
    inplace (bool: True)

    Returns
    -------
    nn.Module
        Activation layer.
    """
    assert (activation is not None)
    if isfunction(activation):
        return activation()
    elif isinstance(activation, str):
        if activation     == "sigmoid":
            return  nn.Sigmoid(inplace=inplace)
        elif activation   == "tanh":
            return nn.Tanh(inplace=inplace)
        elif activation   == "gelu":
            return  nn.GELU()    
        elif activation   == "relu":
            return nn.ReLU(inplace=inplace)
        elif activation == "relu6":
            return nn.ReLU6(inplace=inplace)
        elif activation == "swish":
            return  lambda x: x * torch.sigmoid(x)                      # Swish()  https://arxiv.org/abs/1710.05941.
        elif activation == "hswish":
            return lambda x: x * F.relu6(x+3.0, inplace=inplace) / 6.0  # HSwish() https://arxiv.org/abs/1905.02244.
        elif activation == "hsigmoid":
            return lambda x: F.relu6(x+3.0, inplace=inplace) / 6.0      # HSigmoid() https://arxiv.org/abs/1905.02244.
        
        else:
            raise NotImplementedError()
    else:
        assert isinstance(activation, nn.Module)
        return activation

