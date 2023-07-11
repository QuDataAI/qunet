import math, copy
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn

from ..config  import Config
from .total    import Create, ShiftFeatures

#===============================================================================

class ResidualAlign(nn.Sequential):
    """
    Выравниваем вход и выход блока, если из размерности отличаются
    """
    def __init__(self, Ein, Eout, stride, norm, dim):

        if dim==1: layers = [ nn.Linear(Ein, Eout) ]
        else:      layers = [ nn.Conv2d(Ein, Eout, kernel_size=1, stride=stride, bias=False) ]

        norm = Create.norm(Eout, norm,  dim)
        if type(norm) != nn.Identity:
            layers.append( norm )        
        super().__init__(*layers)

#===============================================================================

class ResidualAfter(nn.Sequential):
    """
    Нормализация и активация после суммирования skip-connection и блок
    """               
    def __init__(self, E, norm, dim, drop_d, shift, fun):
        layers = []
        if norm:
            layers.append( Create.norm(E, norm,  dim) )     
        if drop_d:
            layers.append( Create.dropout(drop_d) )
        if shift:
            layers.append( ShiftFeatures() )
        if fun:
            layers.append(Create.activation(fun))

        super().__init__(*layers)        

#===============================================================================

class Residual(nn.Module):
    """Residual block:
    ```
      ┌─────────────── align(x) ───────────────┐
      │                                        │
      │                                        │
    ──┴─╳── norm(x) ── block(x) ── * ── * ──── + ── norm(x) ── fun(x) ── drop(x) ──
        │                          │    │
      with p drop block        gamma    1+std*randn()   may be missing

    if dim==1: x.shape=(B,T,E)    for transformer, 1d vit
    if dim==2: x.shape=(B,E,H,W)  for 2d vit, ResidualCNN (E=channels)

    align(x) = x                              if y.shape == x.shape for y=block(x)
             = Linear(E_in, E_out)            if y.shape != x.shape and dim==1
             = Conv2d(E_in, E_out, kernel=1)  if y.shape != x.shape and dim==2

    (res==1): gamma = 1; (res==2) gamma - trainable scalar; (res==3) gamma - trainable vector(E)
    ```
    """
    def __init__(self, block, E, Eout=None, stride=1, 
                res=1, gamma=0.1,
                norm_before=1, norm_after=0, norm_align=0, 
                drop_d_after=1, dim=1, shift_after=0, fun_after = "",name=""):
        super().__init__()
        Eout = Eout or E
        self.name = name

        if   norm_before == 1:
            self.norm  = nn.BatchNorm2d(E)    if dim==2 else nn.BatchNorm1d(E)
        elif norm_before == 2:
            self.norm  = nn.InstanceNorm2d(E) if dim==2 else nn.LayerNorm (E)
        else:
            self.norm  = nn.Identity()

        self.block = block
        if   res == 3:                # training multiplier for each components
            if dim==2:
                self.gamma = nn.Parameter( torch.empty(1, E, 1,1).fill_(float(gamma)) )
            else:
                self.gamma = nn.Parameter( torch.empty(1, 1, E ).fill_(float(gamma)) )
        elif res == 2:                # training common multiplier
            self.gamma = nn.Parameter( torch.tensor(float(gamma)) )     # 0 ??? Julius Ruseckas
        else:                         # constant multiplayer
            self.register_buffer("gamma", torch.tensor(float(res)) )  # 1 or ...

        if Eout == E and stride==1:
            self.align = nn.Identity()
        else:
            self.align = ResidualAlign(E, Eout, stride, norm_align, dim)

        if norm_after or drop_d_after or shift_after or fun_after:
            self.after = ResidualAfter(Eout, norm_after, dim,  drop_d_after, shift_after, fun_after)
        else:
            self.after = nn.Identity()

        self.p     = 0        
        self.std   = torch.tensor(float(0.))

        # для статистической информации:
        self.debug_state  = False   # see forward
        self.beta   = 0.9           # value of smoothing
        self.sqr_x  = None          # average of square value of block input
        self.sqr_dx = None

        self.gamma_d  = None        # average for gamma data and grad
        self.gamma_g  = None        # (see update)
        self.index    = 1 if dim==2 else -1

    #---------------------------------------------------------------------------

    def update(self):
        """ Called by trainer before optim.zero_grad() """
        if hasattr(self.block, 'update'):
            self.block.update()

        b, b1 = self.beta, 1-self.beta
        d = self.gamma.detach().square().mean()  # усредняем gammma**2 и её градиент**2
        if self.gamma_d is None: self.gamma_d = d
        else:                    self.gamma_d = b * self.gamma_d + b1 * d

        if self.gamma.requires_grad and self.gamma.grad is not None:
            g = self.gamma.grad.square().mean()
            if self.gamma_g is None: self.gamma_g = g
            else:                    self.gamma_g = b * self.gamma_g + b1 * g

    #---------------------------------------------------------------------------

    def debug(self, value=True, beta=None):
        self.debug_state = value
        if beta is not None:
            self.beta = beta
    #---------------------------------------------------------------------------

    def set_drop(self, drop=None, drop_block=None, std=None, p=None):
        if drop is not None:
            self.drop.p = drop

        if drop_block is not None:
            self.block.set_drop(drop_block)

        if std is not None:
            self.std = torch.tensor(float(std))

        if p is not None:
            self.p = p

    #---------------------------------------------------------------------------

    def forward(self, x):                      # (B,T,E) or (B,E,H,W)
        if self.p and self.p > torch.rand(1) and  type(self.align) is nn.Identity:
            return x                           # skip block (!)

        if self.debug_state:
            dx = self.gamma * self.block( self.norm(x) )
            x  = self.align(x)

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
                    x = self.align(x) + self.gamma * self.block( self.norm(x) ) * (1+torch.randn((x.size(0),x.size(1),1), device=x.device)*self.std)
                else:
                    x = x + dx * (1+torch.randn((x.size(0), 1, x.size(2), x.size(3)), device=x.device)*self.std)
            else:
                x = self.align(x) + self.gamma * self.block( self.norm(x) )

        return self.after(x) 

    #---------------------------------------------------------------------------
