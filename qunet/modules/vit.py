import copy
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn

from ..config     import Config
from .total       import get_activation
from .transformer import Transformer
"""
https://ai.stackexchange.com/questions/28326/why-class-embedding-token-is-added-to-the-visual-transformer
"""

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

class ProjViT(nn.Module):
    """    
    
    """
    def __init__(self,  *args, **kvargs) -> None:
        """ Проектор ViT из картинки в векоры патчей

        Args
        ------------
            input  (tuple=None):
                input tensor shape: (channels, height, width); 
            hidden (int = None)
                number of hidden channles
            E (int = 128):
                patch embedding dimension
            pw, ph (int = 16,16):
                patch size in px
            drop_cnn (float = 0.1): 
                Dropout2d in hidden cnn
            drop_out (float = 0.1):
                out Dropout
        """
        super().__init__()
        self.cfg = Config(
            input = None,   # image (channels, height, width)
            hidden= None,   # is hidden Conv2d
            E        = 256, # patch embedding dimension
            ph       = 8,   # patch height
            pw       = 8,   # patch width            
            drop_cnn = 0.1, # dropout2d in hidden cnn
            drop_out = 0.1, # out drop
        )        
        cfg = self.cfg.set(*args, **kvargs)

        self.n_patches = (cfg.input[1] // cfg.ph) * (cfg.input[2] // cfg.pw)

        if cfg.hidden:
            self.proj = nn.Sequential(
                nn.Conv2d(cfg.input[0], cfg.hidden, 3, padding=1),
                nn.GELU(),
                nn.Dropout2d(cfg.drop_cnn),
                nn.Conv2d(cfg.hidden, cfg.E, kernel_size=(cfg.ph, cfg.pw), stride=(cfg.ph, cfg.pw)) )
        else:
            self.proj = nn.Conv2d(cfg.input[0], cfg.E, kernel_size=(cfg.ph, cfg.pw), stride=(cfg.ph, cfg.pw) )        

        self.pos  = nn.Parameter(torch.zeros(1, self.n_patches, cfg.E))        
        self.drop = nn.Dropout(cfg.drop_out)        


    #---------------------------------------------------------------------------

    def forward(self, x):      # (B,C,H,W)
        x = self.proj(x)       # (B,E,Py,Px)
        x = x.flatten(2)       # (B,E,P)     P = Py * Px
        x = x.transpose(1, 2)  # (B,P,E)    
        x = x + self.pos       #             position encoding
        x = self.drop(x)
        return x               # (B,P,E)

    @staticmethod
    def unit_test():
        B,C,H,W = 1, 3, 64,32
        proj = ProjViT(input=(C,H,W), pw=16, ph=8, E=128, hidden=16)
        x = torch.rand(B,C,H,W)
        y = proj(x)        
        print(f"ok ProjViT: {tuple(x.shape)} -> {tuple(y.shape)}")
        return True
#===============================================================================
