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

class BaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.beta = 0.9
        self.datas = []
        self.grads = []

    #---------------------------------------------------------------------------

    def update(self, in_modules=[]):
        """
        Вызывается тренереном перед обнулением градиентов, если модуль добавлен fit(states=[])
        in_modules = [ nn.Linear, nn.Conv2d]
        """
        i = 0
        for layer in self.mlp:
            if len(in_modules)==0 or type(layer) in in_modules:
                if layer.weight.grad is None:
                    g = 0                    
                else:
                    g = layer.weight.grad.square().mean().sqrt()
                if len(self.grads) == i:
                    self.grads.append(g)                    
                else:
                    self.grads[i] = self.cfg.beta * self.grads[i]  + (1-self.cfg.beta) * g

                d = layer.weight.detach().square().mean().sqrt()
                if len(self.datas) == i:                    
                    self.datas.append(d)                    
                else:
                    self.datas[i] = self.beta * self.datas[i]  + (1-self.beta) * d

                i += 1

    #---------------------------------------------------------------------------                

    def plot(self ):
        """
        Вызывается тренереном при построении графиков модели, если модуль добавлен fit(states=[])        
        """
        pass