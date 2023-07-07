import math, copy
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn

from ..config           import Config
from .mlp               import MLP
from .residual          import Residual
from .transformer_plot  import plot_transformer_blocks

    
#===============================================================================

class Attention(nn.Module):
    """
    Self Attention: (B,T,E) -> (B,T,E)
    base on: https://github.com/karpathy/nanoGPT/blob/master/model.py
    """
    def __init__(self,  *args, **kvargs) -> None:
        """
        Self Attention: (B,T,E) -> (B,T,E)

        Args
        ------------
            E (int):
                tokens embedding dimension
            H (int):
                number of heads (E % H == 0 !)
            drop (float=0.0):
                dropout probability after softmax
            res (int=1):
                kind of skip-connections (for TransformerBlock): (1) x+f(x), (2)  x+w*f(x) - w training one for all E; (3) w training for each E
            skip (float = 1.):
                # fixed multiplier for skip loop:  skip*x + w*f(x) - can be turned off
            causal (bool=False):
                kind of causal attention mask (True: GPT, False: BERT)
            T_max (int=2048):
                maximum number of tokens (needed for causal==True)

        Example
        ------------
        ```
        B, T, E, H = 1, 10, 128, 16
        att = Attention(E=E, H=H)
        x = torch.rand(B,T,E)         # batch, tokens, embedding
        y = att(x)                    # (B,T,E)
        ```
        """
        super().__init__()
        self.cfg = Attention.default()
        self.cfg.set(*args, **kvargs)
        self.create()

        self.beta = 0.9
        self.data = None
        self.grad = None

    #---------------------------------------------------------------------------

    @staticmethod
    def default():
        return copy.deepcopy(Config(
            E      = None,     # размерность эмбедига
            H      = 1,        # число голов (E % H == 0 !)                        
            causal = False,    # причинное внимание (как в GPT) иначе как в BERT
            T_max  = 2048,     # максимальное число токенов (нужно для causal==True)
            drop   = 0,        # dropout after softmax
        ))

    #---------------------------------------------------------------------------

    def create(self):
        cfg = self.cfg
        assert cfg.E is not None and cfg.E > 0 and cfg.H > 0, f"must be E and H in cfg:{cfg.get_str()}"
        assert cfg.E % cfg.H == 0,  "E must be div by H: {cfg.E} % {cfg.H} != 0"

        T,E = cfg.T_max, cfg.E

        self.c_attn = nn.Linear(E,3*E)  # key, query, value projections for all heads
        self.c_proj = nn.Linear(E,  E)  # output projection

        self.drop  = nn.Dropout(cfg.drop)      # regularization        

        if cfg.causal:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(T, T)).view(1,1,T,T))

    #---------------------------------------------------------------------------

    def forward(self, x):
        """ (B,T,E) -> (B,T,E) """
        assert x.ndim == 3, f"wrong input x.ndim = {x.ndim}"
        # batch size, sequence length, embedding dimensionality and number of heads:
        (B,T,E), H = x.size(), self.cfg.H
        assert E == self.cfg.E, f"wrong input {E} != {self.cfg.E}"

        # calc (query, key, values) for all heads and move head forward to be the batch dim
        q,k,v  = self.c_attn(x).split(E, dim=2)
        k = k.view(B,T,H, E // H).transpose(1,2) # (B,H,T, hs) hs = E/nh
        q = q.view(B,T,H, E // H).transpose(1,2) # (B,H,T, hs)
        v = v.view(B,T,H, E // H).transpose(1,2) # (B,H,T, hs)

        # causal self-attention; Self-attend: (B,H,T, hs) x (B,H, hs, T) -> (B,H,T,T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.cfg.causal:  # заполняем верхний угол треугольной матрицы -inf
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

        att = torch.softmax(att, dim=-1)
        att = self.drop(att)

        y = att @ v                          # (B,H,T,T) x (B,H,T,hs) -> (B,H,T,hs)
        y = y.transpose(1,2).contiguous().view(B,T,E) # re-assemble all head side by side
            
        return self.c_proj(y)                   # (B,T,E) output projection

    #---------------------------------------------------------------------------

    def set_drop(self, value):
        self.drop.p = value        
    #---------------------------------------------------------------------------

    def decay(self):
        return set(self.c_attn.weight, self.c_proj.weight)

    #---------------------------------------------------------------------------

    def update(self):        
        d = (self.c_attn.weight.data.square().sum(-1).mean() + self.c_proj.weight.data.square().sum(-1).mean()) / 2
        g = (self.c_attn.weight.grad.square().sum(-1).mean() + self.c_proj.weight.grad.square().sum(-1).mean()) / 2            

        b, b1 = self.beta, 1-self.beta
        if self.data is None: self.data = d
        else:                 self.data = b * self.data + b1 * d
        if self.grad is None: self.grad = g
        else:                 self.grad = b * self.grad + b1 * g
            
    #---------------------------------------------------------------------------

    @staticmethod
    def unit_test():
        B, T, E, H = 1, 10, 128, 16
        att = Attention(E=E, H=H)
        x = torch.rand(B,T,E)
        y = att(x)
        r1 = x.shape == y.shape
        if not r1:
            print(f"!! Attention: x.shape={x.shape} != y.shape={y.shape}")

        cfg = Attention.default()(E=E)
        att = Attention(cfg)
        y = att(x)
        r2 = x.shape == y.shape
        if not r2:
            print(f"!! Attention: x.shape={x.shape} != y.shape={y.shape}")

        res = r1 and r2
        if res:
            print("ok Attention")
        return res

#===============================================================================

class FFT(nn.Module):
    """
    FFT Block from FNet (see: "FNet: Mixing Tokens with Fourier Transforms")
    """
    def __init__(self,  *args, **kvargs) -> None:
        """
        FFT Block from FNet (see: "FNet: Mixing Tokens with Fourier Transforms")

        Args:
        ------------
            after (int = 1):
                after fft2: (1) take Re, (2) take Im, (3) Re**2+Im**2

        Example
        ------------
        ```
        B, T, E = 1, 10, 128
        fft = FFT()
        x = torch.rand(B,T,E)         # batch, tokens, embedding
        y = fft(x)                    # (B,T,E)
        ```
        """
        super().__init__()
        self.cfg = FFT.default()
        self.cfg.set(*args, **kvargs)        

        self.data = torch.tensor(float(0))
        self.grad = torch.tensor(float(0))

    #---------------------------------------------------------------------------

    @staticmethod
    def default():
        return copy.deepcopy(Config(
            after = 1,         # после fft2: 1 - брать Re, 2 брать Im, 3 Re**2+Im**2
        ))

    #---------------------------------------------------------------------------

    def forward(self, x):                      # (B,T,E)
        x = torch.fft.fft2(x)
        if self.cfg.after   == 1:
            x = x.real
        elif self.cfg.after == 2:
            x = x.imag
        else:
            x = x.real.pow(2) + x.imag.pow(2)

        return x                               # (B,T,E)

    #---------------------------------------------------------------------------
    
    def set_drop(self, value):
        pass

    #---------------------------------------------------------------------------

    def decay(self):
        return set()

    #---------------------------------------------------------------------------

    def update(self):
        pass

    #---------------------------------------------------------------------------

    @staticmethod
    def unit_test():
        B, T, E = 1, 10, 128
        fft = FFT()
        x = torch.rand(B,T,E)
        y = fft(x)
        res = x.shape == y.shape
        if not res:
            print(f"!! FFT: x.shape={x.shape} != y.shape={y.shape}")
        else:
            print("ok FFT")
        return res

#===============================================================================

class  TransformerBlock(nn.Module):
    """
    One Transformer Block (it is all you need: fft, attention, mlp)
    """
    def __init__(self,  *args, **kvargs) -> None:
        """
        One Transformer Block (it is all you need: fft, attention, mlp)

        Args
        ------------
            E (int):
                tokens embedding dimension
            H (int=1):
                number of heads (E % H == 0 !)
            drop (float=0.0):
                dropout probability after block
            res (int=1):
                kind of skip-connections: (0) f(x) - none; (1) x+f(x), (2)  x*w+f(x) training one for all E; (3) training for each E
            causal (bool=False):
                kind of causal attention mask (True: GPT, False: BERT)
            T_max (int=2048):
                maximum number of tokens (needed for causal==True)
            gamma (float = 0.1):
                initial value of the learning residual multiplier

        Example
        ------------
        ```
        B, T, E, H = 1, 10, 128, 16
        block = TransformerBlock(E=E, H=H)
        x = torch.rand(B,T,E)
        y = block(x)
        ```
        """
        super().__init__()
        self.cfg = TransformerBlock.default()
        cfg = self.cfg.set(*args, **kvargs)

        self.cfg.att.E = self.cfg.mlp.input = self.cfg.mlp.output = cfg.E

        if 'H' in kvargs:
            self.cfg.att.H = kvargs['H']

        if 'drop' in kvargs:
            self.cfg.att.drop = self.cfg.mlp.drop  = kvargs['drop']

        if 'causal' in kvargs:
            self.cfg.att.causal = kvargs['causal']

        if 'T_max'  in kvargs:
            self.cfg.att.T_max = kvargs['T_max']

        self.create()

    #---------------------------------------------------------------------------

    @staticmethod
    def default():
        return copy.deepcopy(Config(
            E      = None,              # tokens embedding dimension    

            is_fft = 0,
            is_att = 1,
            is_mlp = 1,            

            res    = 2,
            norm   = 2,
            skip   = 1.0,
            gamma  = 0.0,
            drop   = 0.0,               # dropout probability after block

            att = Attention.default(),
            mlp = Config(MLP.default(), stretch=4, res=1, skip=1.0),
            fft = FFT.default(),
        ))

    #---------------------------------------------------------------------------

    def create(self):
        cfg = self.cfg
        assert cfg.E is not None and cfg.E > 0,  f"TransformerBlock: embedding needs to be defined: E={cfg.E}, input={cfg.mlp.input}, output={cfg.mlp.output}"

        layers = []
        if cfg.is_fft:
            layers.append( Residual(FFT(cfg.fft),       E=cfg.att.E, res=cfg.res, skip=cfg.skip, gamma=cfg.gamma, drop=cfg.drop, norm_before=cfg.norm, name="fft") )
        if cfg.is_att:
            layers.append( Residual(Attention(cfg.att), E=cfg.att.E, res=cfg.res, skip=cfg.skip, gamma=cfg.gamma, drop=cfg.drop, norm_before=cfg.norm, name="att") )
        if cfg.is_mlp:
            layers.append( Residual(MLP(cfg.mlp),       E=cfg.att.E, res=cfg.res, skip=cfg.skip, gamma=cfg.gamma, drop=cfg.drop, norm_before=cfg.norm, name="mlp") )
    
        self.layers = nn.Sequential( *layers )

    #---------------------------------------------------------------------------

    def forward(self, x):
        """
        (B,T,E) -> (B,T,E)
        """
        return self.layers(x)                                            # (B,T,E)

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
    #---------------------------------------------------------------------------

    @staticmethod
    def unit_test():
        B, T, E, H = 1, 10, 128, 16
        block = TransformerBlock(E=E, H=H)
        x = torch.rand(B,T,E)
        y = block(x)
        res = x.shape == y.shape
        if not res:
            print(f"!! TransformerBlock: x.shape={x.shape} != y.shape={y.shape}")
        else:
            print("ok TransformerBlock")
        return res

#===============================================================================

class  Transformer(nn.Module):
    """
    Transformer is all you need (fft, attention, mlp)
    """
    def __init__(self,  *args, **kvargs) -> None:
        """
        Transformer is all you need (fft, attention, mlp)

        Args
        ------------
            E (int):
                tokens embedding dim
            H (int):
                number of heads E % H == 0 !
            n_blocks (int=1):
                number of transformer blocks
            drop: (float=0.0):
                dropout in attention, mlp and after res block
            res (int=1):
                kind of skip-connections: (0) f(x) - none; (1) x+f(x), (2)  x*w+f(x) training one for all E; (3) training for each E
            causal (bool=False):
                kind of causal attention mask (True: GPT, False: Bert)
            T_max  (int=2048):
                maximum number of tokens (needed for causal==True)
            is_mix (int=0):
                will be
            is_fft (int=0):
                there is a FFT (FNet) block, can be a list (for each block)
            is_att  (int=1):
                there is an attention block, can be a list (for each block)
            is_mlp  (int=1):
                there is an MLP block, can be a list (for each block)
            gamma (float = 0.1):
                initial value of the learning residual multiplier
        Example:
        ------------
        ```
        m = Transformer(E=64, H=8, n_blocks=3, is_fft=[1,1,0], is_att=[0,0,1])

        x = torch.rand(1,10,64)  # (B,T,E) = (batch, token, embedding)
        y = m(x)                 # (B,T,E) -> (B,T,E)
        ```
        """
        super().__init__()
        self.cfg = Transformer.default()
        cfg = self.cfg.set(*args, **kvargs)

        cfg.block.E     = cfg.E
        cfg.block.res   = cfg.res
        cfg.block.gamma = cfg.gamma
        cfg.block.skip  = cfg.skip
        
        H = kvargs['H'] if 'H' in kvargs else (self.cfg.H if self.cfg.has('H') else self.cfg.block.att.H)
        self.cfg.block.att.H = H

        res = kvargs['res'] if 'res' in kvargs else (self.cfg.res if self.cfg.has('res') else self.cfg.block.att.res)
        self.cfg.block.fft.res = self.cfg.block.att.res = self.cfg.block.mlp.res = res

        drop = kvargs['drop'] if 'drop' in kvargs else (self.cfg.drop if self.cfg.has('drop') else self.cfg.block.att.drop)
        self.cfg.block.fft.drop = self.cfg.block.att.drop = self.cfg.block.mlp.drop = drop

        causal = kvargs['causal'] if 'causal' in kvargs else (self.cfg.causal if self.cfg.has('causal') else self.cfg.block.att.causal)
        self.cfg.block.att.causal = causal

        T_max = kvargs['T_max'] if 'T_max' in kvargs else (self.cfg.T_max if self.cfg.has('T_max') else self.cfg.block.att.T_max)
        self.cfg.block.att.T_max = T_max

        after = kvargs['after'] if 'after' in kvargs else (self.cfg.after if self.cfg.has('after') else self.cfg.block.fft.after)
        self.cfg.block.fft.after = after

        self.create()

    #---------------------------------------------------------------------------

    @staticmethod
    def default():
        return copy.deepcopy(Config(
            E         = None,  # эмбеддинг
            n_blocks  = 1,     # число слоёв трансформера
            is_fft    = 0,     # there is a FFT (FNet) block, can be a list (for each block)
            is_att    = 1,     # there is an attention block, can be a list (for each block)
            is_mlp    = 1,     # there is an MLP block, can be a list (for each block)
            res       = 2,
            skip      = 1.0,
            gamma     = 0.1,   # initial value of the learning residual multiplier
            block = TransformerBlock.default()
        ))

    #---------------------------------------------------------------------------

    def create(self):
        cfg = self.cfg
        is_fft = [cfg.is_fft] * cfg.n_blocks  if type(cfg.is_fft) is int else cfg.is_fft
        is_att = [cfg.is_att] * cfg.n_blocks  if type(cfg.is_att) is int else cfg.is_att
        is_mlp = [cfg.is_mlp] * cfg.n_blocks  if type(cfg.is_mlp) is int else cfg.is_mlp

        assert type(is_fft) is list and type(is_att) is list and type(is_mlp) is list, "Transformer: is_fft, is_att, is_mlp should be int or list of int"
        assert len(is_fft) == len(is_att) and len(is_fft) == len(is_mlp), "Transformer: if is_fft, is_att, is_mlp are lists, their length should be the same"

        blocks = []
        for i in range(cfg.n_blocks):
            block = TransformerBlock(cfg.block, is_fft=is_fft[i], is_att=is_att[i], is_mlp=is_mlp[i])
            blocks.append(block)
        self.blocks= nn.ModuleList(blocks)

    #---------------------------------------------------------------------------

    def forward(self, x):
        """  (B,T,E) -> (B,T,E) """
        for block in self.blocks:
            x = block(x)                           # (B,T,E)
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

    def set_drop(self, fft=0, att=0, mlp=0):
        for block in self.blocks:
            block.set_drop(fft=fft, att=att, mlp=mlp)

    #---------------------------------------------------------------------------

    def plot(self, w=12, h=3, eps=1e-8, bar_width = 0.25, info=False):
        plot_transformer_blocks(self.blocks, w=w, h=h, eps=eps, bar_width = bar_width, info=info)
        
    #---------------------------------------------------------------------------

    @staticmethod
    def unit_test():
        B, T, E, H = 1, 10, 128, 16
        cfg = Config(E=E, H=H, n_blocks=10, is_fft=1, is_att=1, res=2, gamma=0.2)
        m = Transformer(cfg)
        x = torch.rand(B,T,E)  # (B,T,E) = (batch, token, embedding)

        m.debug(True)        
        y = m(x)               # (B,T,E) -> (B,T,E)
        y.mean().backward()
        m.update()
        #print( m.plot(info=True) )

        #for block in m.blocks: print(block.cfg)

        res = x.shape == y.shape
        if not res:
            print(f"!! Transformer: x.shape={x.shape} != y.shape={y.shape}")
        else:
            print(f"ok Transformer: {tuple(x.shape)} -> {tuple(y.shape)}")

        """
        cfg = Config(E=E, H=H, n_blocks=5, is_fft=[1,1,0,0,0], is_att=[0,0,1,1,1])
        m = Transformer(cfg)
        x = torch.rand(B,T,E)  # (B,T,E) = (batch, token, embedding)
        y = m(x)               # (B,T,E) -> (B,T,E)
        print(f"ok Transformer: {tuple(x.shape)} -> {tuple(y.shape)}")
        """
        return res
