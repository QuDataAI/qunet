from inspect import isfunction
import torch, torch.nn as nn
import torch.nn.functional as F
#===============================================================================

class LayerNormChannels(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):           # (B,C,H,W,...)
        x = x.transpose(1, -1)      # (B,W,H,...,C)
        x = self.norm(x)            # avg on index c, 2*C parameters
        x = x.transpose(-1, 1)      # (B,C,H,W,...)
        return x

#===============================================================================

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
            return  nn.Sigmoid()
        elif activation   == "tanh":
            return nn.Tanh()
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

#===============================================================================


def get_norm(E,  norm, dim):
    if norm == 1:
        return nn.BatchNorm1d(E)    if dim==1 else nn.BatchNorm2d(E) 
    if norm == 2:
        return nn.LayerNorm (E)     if dim==1 else LayerNormChannels(E)
    if norm == 3:
        return nn.InstanceNorm1d(E) if dim==1 else nn.InstanceNorm2d(E)
    return nn.Identity()

#===============================================================================

def get_model_layers(model, kind, layers=[]):        
    """Get a list of model layers of type kind

    Example:
    ----------
    layers = get_model_layers(model, (nn.Dropout1d, nn.Dropout2d) )
    """
    for mo in model.children():
        if isinstance(mo, kind):
            layers.append(mo)
        else:
            layers = get_model_layers(mo, kind, layers=layers)
    return layers
