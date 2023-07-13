import math, copy
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn

from ..config  import Config
from .total    import Create, ShiftFeatures, Scaler

#===============================================================================

class SkipConnection(nn.Sequential):
    """
    Выравниваем вход и выход блока, если из размерности отличаются
    """
    def __init__(self, Ein, Eout, norm, stride=1, dim=1):                
        layers = []
        if Ein != Eout or stride > 1:
            assert dim in [1,2], f"Wrong dim={dim} in SkipConnection  {Ein}->{Eout}, stride={stride}"

            if   dim==1: layers.append( nn.Linear(Ein, Eout, bias=False) )
            elif dim==2: layers.append( nn.Conv2d(Ein, Eout, kernel_size=1, stride=stride, bias=False) )

            if norm:
                layers.append( Create.norm(Eout, norm,  dim) )
        
        super().__init__(*layers)
#===============================================================================

class AfterResidual (nn.Sequential):
    """
    Нормализация и активация после суммирования skip-connection и блок
    """               
    def __init__(self, E, norm, dim, drop, shift, fun):
        
        layers = []
        if norm:
            layers.append( Create.norm(E, norm,  dim) )     
        if drop:
            layers.append( Create.dropout(drop) )
        if shift:
            layers.append( ShiftFeatures() )
        if fun:
            layers.append(Create.activation(fun))                

        super().__init__(*layers)

#===============================================================================

class Residual(nn.Module):
    """Residual block:
    ```
     ┌─────────────── align(x) ── n(x) ───────┐
     │                                        │
     │                                        │
    ─┴─╳── norm(x) ── block(x) ── n(x) ─ * ── * ──── + ── n(x) ── f(x) ── drop(x) ──
       │                                 │      │
      with p drop block              scale      1+std*randn() 

    if dim==1: x.shape=(B,T,E)    for transformer, 1d vit
    if dim==2: x.shape=(B,E,H,W)  for 2d vit, ResidualCNN (E=channels)

    align(x) = x                              if y.shape == x.shape for y=block(x)
             = Linear(E_in, E_out)            if y.shape != x.shape and dim==1
             = Conv2d(E_in, E_out, kernel=1)  if y.shape != x.shape and dim==2

    (mult=0): нет; (mult=1) scale - trainable scalar; (mult=2) scale - trainable vector(E)
    ```
    """
    def __init__(self, block, E, Eout=None, stride=1, 
                mult=0, scale=1.,
                norm_before=1, norm_after=0, norm_align=0, norm_block=0,
                drop_after=1, dim=1, shift_after=0, fun_after = "",name=""):
        super().__init__()
        Eout = Eout or E
        self.name = name
        
        self.norm_before = Create.norm(E,    norm_before,  dim)        
        self.block = block
        self.norm_block  = Create.norm(Eout, norm_block,   dim)

        self.scaler = Scaler(kind=mult, value=scale, dim=dim, E=E) if mult else nn.Identity()

        self.align = SkipConnection (E, Eout, norm_align, stride, dim)        
        self.after = AfterResidual (Eout, norm_after, dim,  drop_after, shift_after, fun_after)
        if len(self.after) == 0:
            self.after = nn.Identity()

        self.p     = None  if (self.align) else 0.0
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

        if type(self.scaler) is nn.Identity or self.scaler.scale is None:
            return
        
        b, b1 = self.beta, 1-self.beta
        d = self.scaler.scale.detach().square().mean()  # усредняем gammma**2 и её градиент**2
        if self.gamma_d is None: self.gamma_d = d
        else:                    self.gamma_d = b * self.gamma_d + b1 * d

        if self.scaler.scale.requires_grad and self.scaler.scale.grad is not None:
            g = self.scaler.scale.grad.square().mean()
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

    def set_drop_block(self, p=None):
        if p is not None and self.p is not None:
            self.p = p

    #---------------------------------------------------------------------------

    def forward(self, x):                      # (B,T,E) or (B,E,H,W)
        if self.p and self.p > torch.rand(1) and  type(self.align) is nn.Identity:
            return x                           # skip block (!)

        if self.debug_state:
            dx = self.scaler ( self.norm_block( self.block( self.norm_before(x) ) ) )
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
                    x = self.align(x) + self.scaler ( self.norm_block( self.block( self.norm_before(x) ) ) ) * (1+torch.randn((x.size(0),x.size(1),1), device=x.device)*self.std)
                else:
                    x = x + dx * (1+torch.randn((x.size(0), 1, x.size(2), x.size(3)), device=x.device)*self.std)
            else:
                x = self.align(x) + self.scaler ( self.norm_block( self.block( self.norm_before(x) ) ) )

        return self.after(x) 

    #---------------------------------------------------------------------------
