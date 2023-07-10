import copy
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn

from .residual     import Residual
from ..config     import Config
from .transformer_plot  import plot_transformer_blocks

"""
https://ai.stackexchange.com/questions/28326/why-class-embedding-token-is-added-to-the-visual-transformer
"""
    
#===============================================================================

class VitEmb(nn.Module):
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
            patch (tuple = (8,8):
                patch height and width in px
            drop_cnn (float = 0.): 
                Dropout2d in hidden cnn
            drop_out (float = 0.):
                out Dropout
        """
        super().__init__()
        self.cfg = Config(
            input    = None,    # image (channels, height, width)
            hidden   = None,    # is hidden Conv2d
            E        = 256,     # patch embedding dimension
            patch    = (8,8),   # patch (height, width)              
            drop     = 0.0,     # dropout2d in hidden cnn            
            flat     = True,
        )        
        cfg = self.cfg.set(*args, **kvargs)
        
        # number of patches on y and x
        self.grid      = (cfg.input[1] // cfg.patch[0],   cfg.input[2] // cfg.patch[1])
        self.n_patches = self.grid[0] * self.grid[1]

        if cfg.hidden:
            self.emb = nn.Sequential(
                nn.Conv2d(cfg.input[0], cfg.hidden, 3, padding=1),
                nn.GELU(),
                nn.Dropout2d(cfg.drop),
                nn.Conv2d(cfg.hidden, cfg.E, kernel_size=cfg.patch, stride=cfg.patch) )
        else:
            self.emb = nn.Conv2d(cfg.input[0], cfg.E, kernel_size = cfg.patch, stride=cfg.patch)        

        if cfg.flat:
            self.flat = nn.Flatten(2)
            self.pos  = nn.Parameter( 0.02 * torch.randn(1, self.n_patches, cfg.E) )                    
            self.drop = nn.Dropout(cfg.drop)        
        else:            
            self.flat = None
            self.pos  = nn.Parameter( 0.02 * torch.randn(1, cfg.E, *self.grid) )                    
            self.drop = nn.Dropout2d(cfg.drop)        

    #---------------------------------------------------------------------------

    def forward(self, x):            # (B,C,H,W)
        x = self.emb(x)              # (B,E,nx,ny)

        if self.flat is not None:
            x = self.flat(x)         # (B,E,P)     P = ny*nx
            x = x.transpose(1, 2)    # (B,P,E)    

        x = x + self.pos             #             position encoding
        x = self.drop(x)
        return x                     # (B,P,E)

    #---------------------------------------------------------------------------

    @staticmethod
    def unit_test():
        B,C,H,W = 1, 3, 64,32
        emb = VitEmb(input=(C,H,W), patch=(16, 8), E=128, hidden=16, flat=False)
        x = torch.rand(B,C,H,W)
        y = emb(x)                
        print(f"ok VitEmb: {tuple(x.shape)} -> {tuple(y.shape)}")
        return True

#===============================================================================
#                                      ViT 2D
#===============================================================================


class Attention2d(nn.Module):
    """
    Attention with relative position encoding  (B,E,ny,nx) -> (B,E,ny,nx)
    https://juliusruseckas.github.io/ml/cifar10-vit.html    
    """
    def __init__(self, E, H, grid):
        """
        E - embedding; H - number of heads
        shape = (ny,nx)
        """
        super().__init__()
        self.heads = H
        self.head_channels = E // H
        self.scale = self.head_channels**-0.5
        
        self.to_keys    = nn.Conv2d(E, E, kernel_size=1)  
        self.to_queries = nn.Conv2d(E, E, kernel_size=1)
        self.to_values  = nn.Conv2d(E, E, kernel_size=1)
        self.unifyheads = nn.Conv2d(E, E, kernel_size=1)
        
        ny, nx = grid
        self.pos_enc = nn.Parameter(0.02*torch.randn(self.heads, (2*ny - 1) * (2*nx - 1)))
        self.register_buffer("relative_indices", self.get_indices(ny, nx))

    #---------------------------------------------------------------------------

    def forward(self, x):
        b, _, ny, nx = x.shape              # B,E,ny,nx
        
        keys    = self.to_keys(x).   view(b, self.heads, self.head_channels, -1)
        values  = self.to_values(x). view(b, self.heads, self.head_channels, -1)
        queries = self.to_queries(x).view(b, self.heads, self.head_channels, -1)
        
        att = keys.transpose(-2, -1) @ queries
        
        indices     = self.relative_indices.expand(self.heads, -1)
        rel_pos_enc = self.pos_enc.gather(-1, indices)
        rel_pos_enc = rel_pos_enc.unflatten(-1, (ny * nx, ny * nx))
        
        att = att * self.scale + rel_pos_enc
        att = nn.functional.softmax(att, dim=-2)
        
        out = values @ att
        out = out.view(b, -1, ny, nx)
        out = self.unifyheads(out)
        return out

    #---------------------------------------------------------------------------

    @staticmethod
    def get_indices(ny, nx):
        y = torch.arange(ny, dtype=torch.long)
        x = torch.arange(nx, dtype=torch.long)
        
        y1, x1, y2, x2 = torch.meshgrid(y, x, y, x, indexing='ij')
        indices = (y1 - y2 + ny - 1) * (2 * nx - 1) + x1 - x2 + nx - 1
        indices = indices.flatten()
        
        return indices

    #---------------------------------------------------------------------------
    
    def update(self):
        pass

    #---------------------------------------------------------------------------
    
    def decay(self):
        return set( self.to_keys.weight, self.to_queries.weight, self.to_values.weight,
                   self.unifyheads.weight)

#===============================================================================

class MLP2d(nn.Sequential):
    def __init__(self, E, stretch=4):
        hidden_channels = E * stretch
        super().__init__(
            nn.Conv2d(E, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, E, kernel_size=1)   
        )
    #---------------------------------------------------------------------------
    
    def update(self):
        pass

    #---------------------------------------------------------------------------

    def decay(self):
        return set( self[0].weight, self[-1].weight)
    
#===============================================================================

class ViT2dBlock(nn.Module):
    def __init__(self, E, H, grid,  res=1):
        super().__init__()
        self.layers = nn.Sequential(
            Residual(                
                Attention2d(E,H, grid),                
                E = E,  dim=2, res=res, name="att"
            ),
            Residual(                                        
                MLP2d(E),                
                E = E,  dim=2, res=res, name="mlp"
            )
        )
    
    def forward(self, x):
        return self.layers(x)

   #---------------------------------------------------------------------------

    def update(self):
        for layer in self.layers:        
            layer.update()

    #---------------------------------------------------------------------------

    def decay(self):
        res = set()
        for layer in self.layers:        
            res.update( layer.decay() )
        return res

    #---------------------------------------------------------------------------

    def set_drop(self, drop):
        for layer in self.layers:        
            layer.set_drop( drop )


    #---------------------------------------------------------------------------

    def debug(self, value=True, beta=None):
        for layer in self.layers:        
            layer.debug = value
            if beta is not None:
                layer.beta = beta

#===============================================================================

class Vit2d(nn.Module):
    def __init__(self,  *args, **kvargs) -> None:
        super().__init__()
        self.cfg = Vit2d.default()
        cfg = self.cfg.set(*args, **kvargs)

        self.emb = VitEmb(Config(input=cfg.input, E=cfg.E, patch=cfg.patch,  
                                 hidden=cfg.hidden, flat=False) )

        blocks = []
        for i in range(cfg.n_blocks):
            block = ViT2dBlock(E=cfg.E, H=cfg.H, grid=self.emb.grid,  res=cfg.res)
            blocks.append(block)
        self.blocks= nn.ModuleList(blocks)        

 #---------------------------------------------------------------------------

    @staticmethod
    def default():
        return copy.deepcopy(Config(
            input    = None,  # (C,H,W) - shape of input image
            hidden   = None,
            E        = None,  # embedding of patches
            H        = 1,     # number of heads
            patch    = None,  # patch size (height, width)                 
            n_blocks = 1,     # число слоёв трансформера
            gamma    = 0.1,   # initial value of the learning residual multiplier  
            res      = 2,
            drop_emb = 0,          
            drop_res = 0.
        ))
    #---------------------------------------------------------------------------

    def forward(self, x):
        """  (B,E,H,W) -> (B,E,ny,bx) """
        x = self.emb(x)                            # (B,E,Ny,Nx)

        for block in self.blocks:
            x = block(x)                           # (B,E,Ny,Nx)
        return x

    #---------------------------------------------------------------------------

    def update(self):
        for block in self.blocks:
            block.update()

    #---------------------------------------------------------------------------

    def decay(self):
        res = set()
        for block in self.blocks:
            res.update(block.decay() )    
        return res        
    
    #---------------------------------------------------------------------------

    def debug(self, value=True, beta=None):
        for block in self.blocks:
            block.debug(value, beta)    

    #---------------------------------------------------------------------------

    def plot(self, w=12, h=3, eps=1e-8, bar_width = 0.25, info=False):
        plot_transformer_blocks(self.blocks, w=w, h=h, eps=eps, bar_width = bar_width, info=info)
        
    #---------------------------------------------------------------------------

    @staticmethod
    def unit_test():
        B,C,H,W = 100, 3, 32,32
        vit = Vit2d(input=(C,H,W), n_blocks=8, patch=(4, 4), E=128, hidden=32, res=2)
        x = torch.rand(B,C,H,W)
        vit.debug(True)        
        vit.update()        
        y = vit(x) 
        print(y.mean())
        #print(y)
        y.mean().backward()
        #vit.update()
        #vit.plot(info=True)


        print(f"ok Vit2d: {tuple(x.shape)} -> {tuple(y.shape)}")
        return True
