import math, copy
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn

from ..config  import Config

#===============================================================================

class LayerNormChannels(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
    
    def forward(self, x):           # (B,C,H,W)
        x = x.transpose(1, -1)      # (B,W,H,C)
        x = self.norm(x)         
        x = x.transpose(-1, 1)      # (B,C,H,W)
        return x
    
#===============================================================================

class Residual(nn.Module):
    """
    Residual block

    Классический highway: w=sigmoid(linear(x)); x*w + (1-w)*mlp, ...
    не работал для нейтрино. Возможно это негативный эффект постоянного
    умножения x (и потом градиента) на число меньшее 1:  x*0.5*0.5*...
    Не работал также linear(x) + mlp, с начальной единичной матрицей :(

    Карпатов сделал ln на входе, хотя многие рисуют на выходе.
    Это сказывается тоько на первом и последнем блоке.
    Наверное как у Карпатова правильнее (нормируем перед вычислениями).
    """
    def __init__(self, module, E, res = 1, skip=1., gamma=0.1, drop=0., vit2d=False, name=""):
        super().__init__()
        self.name = name
        self.norm  = LayerNormChannels(E) if vit2d else nn.LayerNorm (E)
        self.module = module
        if   res == 3:                # training multiplier for each components
            if  vit2d:
                self.gamma = nn.Parameter( torch.empty(1, E, 1,1).fill_(float(gamma)) )
            else:
                self.gamma = nn.Parameter( torch.empty(1, 1, E ).fill_(float(gamma)) )

        elif res == 2:                # training common multiplier
            self.gamma = nn.Parameter( torch.tensor(float(gamma)) )     # 0 ??? Julius Ruseckas
        else:                         # constant multiplayer
            self.register_buffer("gamma", torch.tensor(1.) )

        self.register_buffer("skip", torch.tensor(float(skip)) )
        
        self.drop  = nn.Dropout2d(drop)  if vit2d else nn.Dropout(drop)

        self.debug  = False         # see forward
        self.beta   = 0.9           # value of smoothing
        self.sqr_x  = None          # average of square value of block input
        self.sqr_dx = None
        self.std    = torch.tensor(float(0))

        self.gamma_d  = None        # average for gamma data and grad
        self.gamma_g  = None        # (see update)
        self.index    = 1 if vit2d else -1

    #---------------------------------------------------------------------------

    def update(self):
        """ Call trainer before optim.zero_grad() """
        self.module.update()

        b, b1 = self.beta, 1-self.beta
        d = self.gamma.detach().square().mean()
        if self.gamma_d is None: self.gamma_d = d
        else:                    self.gamma_d = b * self.gamma_d + b1 * d

        if self.gamma.requires_grad and self.gamma.grad is not None:
            g = self.gamma.grad.square().mean()
            if self.gamma_g is None: self.gamma_g = g
            else:                    self.gamma_g = b * self.gamma_g + b1 * g

    #---------------------------------------------------------------------------

    def set_drop(self, value):
        self.drop.p = value

    #---------------------------------------------------------------------------

    def forward(self, x):                 # (B,T,E)
        if self.debug:
            dx = self.gamma * self.module( self.norm(x) )
            x  = self.skip  * x

            v_x  = x. detach().square().sum(self.index).mean()
            v_dx = dx.detach().square().sum(self.index).mean()

            b, b1 = self.beta, 1-self.beta
            if self.sqr_x  is None: self.sqr_x  = v_x
            else:                   self.sqr_x  = b * self.sqr_x  + b1 * v_x
            if self.sqr_dx is None: self.sqr_dx = v_dx
            else:                   self.sqr_dx = b * self.sqr_dx + b1 * v_dx

            if self.std > 0:
                if self.index < 0:
                    x = x + dx * (1+torch.randn((x.size(0),x.size(1),1), device=x.device)*self.std)
                else: # vit 2d
                    x = x + dx * (1+torch.randn((x.size(0),1,x.size(2),x.size(3)), device=x.device)*self.std)
            else:
                x = x + dx
        else:
            if self.std > 0:
                if self.index < 0:
                    x = self.skip * x + self.gamma * self.module( self.norm(x) ) * (1+torch.randn((x.size(0),x.size(1),1), device=x.device)*self.std)
                else:
                    x = x + dx * (1+torch.randn((x.size(0), 1, x.size(2), x.size(3)), device=x.device)*self.std)
            else:
                x = self.skip * x + self.gamma * self.module( self.norm(x) )

        return self.drop(x)
