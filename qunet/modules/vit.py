import copy
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn

from ..config     import Config
from .total       import get_activation
from .transformer import Transformer

#===============================================================================

class ProjViT(nn.Module):
    """    
    """
    def __init__(self,  *args, **kvargs) -> None:
        """ Проектор ViT из картинки в векоры патчей

        Args
        ------------
            input  (None):
                input tensor shape: (channels, height, width); 
            E (int = 128):
                patch embedding dimension
            pw, ph (int = 16,16)
                patch size in px
        """
        super().__init__()
        self.cfg = Config(
            input = None,   # image (channels, height, width)
            E     = 128,    # patch embedding dimension
            ph     = 16,     # patch height
            pw     = 16,     # patch width            
        )
        self.cfg.check_variable_existence = False
        cfg = self.cfg.set(*args, **kvargs)

        self.n_patches = (cfg.input[1] // cfg.ph) * (cfg.input[2] // cfg.pw)
        self.proj = nn.Conv2d(cfg.input[0], cfg.E, kernel_size=(cfg.ph, cfg.pw), stride=(cfg.ph, cfg.pw) )        

    #---------------------------------------------------------------------------

    def forward(self, x):      # (B,C,H,W)
        x = self.proj(x)       # (B,E,Py,Px)
        x = x.flatten(2)       # (B,E,P)    P = Py * Px
        x = x.transpose(1, 2)  # (B,P,E)        
        return x

#===============================================================================

class ViT(nn.Module):
    """    
    """
    def __init__(self,  *args, **kvargs) -> None:
        """Visual Transformer

        Args:
        ------------
            input  (None):
                input tensor shape: (channels, height, width); 
            E (int = 128):
                patch embedding dimension
            pw, ph (int = 16,16):
                patch size in px
            p_drop (float = 0.2):
                dropout after embediing
            pool (bool = True):
                return (B,E) - vector of image features, else (B,P,E) - embedding of each patch

            H (int):
                number of transformer heads E % H == 0 !
            n_blocks (int=1):
                number of transformer blocks
            drop: (float=0.0):
                dropout in attention and mlp
            res (int=1):
                kind of skip-connections: (0) f(x) - none; (1) x+f(x), (2)  x*w+f(x) training one for all E; (3) training for each E

            is_att  (int=1):
                there is an attention block, can be a list (for each block)
            is_mlp  (int=1):
                there is an MLP block, can be a list (for each block)            
                
            is_mix (int=0):
                will be
            is_fft (int=0):
                there is a FFT (FNet) block, can be a list (for each block)

        Example
        ------------
        ```
        B,C,H,W = 1, 3, 64,32
        vit = ViT(input=(C,H,W), pw=16, ph=8, E=128, H=16, n_blocks=5)
        mlp = MLP(E, 10)

        x = torch.rand(B,C,H,W)
        y = vit(x)                # (B,C,H,W) -> (B,E)
        y = mlp(y)                # (B,10)
        ```
        """
        super().__init__()
        self.cfg = Config(            
            input = None,   # image (channels, height, width)
            E     = 128,    # patch embedding dimension
            ph     = 16,    # patch height
            pw     = 16,    # patch width            
            p_drop = 0.2,            
            pool   = True,        

            H        = 1,    
            n_blocks = 1,
            is_fft    = 0,     # there is a FFT (FNet) block, can be a list (for each block)
            is_att    = 1,     # there is an attention block, can be a list (for each block)
            is_mlp    = 1,     # there is an MLP block, can be a list (for each block)            
        )        
        cfg = self.cfg.set(*args, **kvargs)        
        
        self.proj     = ProjViT( *args, **kvargs)
        self.pos      = nn.Parameter(torch.zeros(1, self.proj.n_patches, cfg.E))        
        self.pos_drop = nn.Dropout(p=cfg.p_drop)        
        self.transf   = Transformer(*args, **kvargs)        
        self.norm     = nn.LayerNorm(cfg.E, eps=1e-6)

    #---------------------------------------------------------------------------

    def forward(self, x):               # (B,C,H,W)
        x = self.proj(x)                # (B,P,E)  
        x = x + self.pos
        x = self.pos_drop(x)
        x = self.transf(x)              # (B,P,E)  
        x = self.norm(x)
        if self.cfg.pool:
            x = x.mean(1)               # (B,E)            
        return x
    
    #---------------------------------------------------------------------------
    
    @staticmethod
    def unit_test():
        B,C,H,W = 1, 3, 64,32
        vit = ViT(input=(C,H,W), pw=16, ph=8, E=128, H=16, n_blocks=5, pool=True)
        x = torch.rand(B,C,H,W)
        y = vit(x)        
        print(f"ok ViT: {tuple(x.shape)} -> {tuple(y.shape)}")
        return True

    #---------------------------------------------------------------------------        
    