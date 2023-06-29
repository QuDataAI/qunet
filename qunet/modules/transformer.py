﻿import math, copy
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn

from ..config  import Config
from .mlp      import MLP

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
    def __init__(self, module, dim, res = 1, skip=1.):
        super().__init__()
        self.module  = module
        if   res == 3:                # training multiplier for each components
            self.gamma = nn.Parameter( torch.zeros( dim ) )
        elif res == 2:                # training common multiplier
            self.gamma = nn.Parameter( torch.zeros(1) )     # 0 ??? Julius Ruseckas
        else:                         # constant multiplayer
            self.register_buffer("gamma", torch.tensor(1.) )

        self.register_buffer("skip", torch.tensor(float(skip)) )

        self.norm  = nn.LayerNorm (dim)

        self.debug  = False         # see forward
        self.beta   = 0.9           # value of smoothing
        self.sqr_x  = None          # average of square value of block input
        self.sqr_dx = None
        self.std    = torch.tensor(float(0))

        self.gamma_d   = None       # average for gamma data and gard
        self.gamma_g   = None       # (see update)

    #---------------------------------------------------------------------------

    def update(self):
        """ Call trainer before optim.zero_grad() """
        self.module.update()

        if self.gamma.requires_grad and self.grad is not None:
            d = self.gamma.detach().square().mean()
            g = self.gamma.gard    .square().mean()

            b, b1 = self.beta, 1-self.beta
            if self.gamma_d is None: self.gamma_d = d
            else:                    self.gamma_d = b * self.gamma_d + b1 * d
            if self.grad is None:    self.gamma_g = g
            else:                    self.gamma_g = b * self.gamma_g + b1 * g

    #---------------------------------------------------------------------------

    def forward(self, x):                 # (B,T,E)
        if self.debug:
            dx = self.gamma * self.module( self.norm(x) )
            x  = self.skip  * x

            v_x  = x. detach().square().sum(-1).mean()
            v_dx = dx.detach().square().sum(-1).mean()

            b, b1 = self.beta, 1-self.beta
            if self.sqr_x  is None: self.sqr_x  = v_x
            else:                   self.sqr_x  = b * self.sqr_x  + b1 * v_x
            if self.sqr_dx is None: self.sqr_dx = v_dx
            else:                   self.sqr_dx = b * self.sqr_dx + b1 * v_dx

            if self.std > 0:
                return x + dx * (1+torch.randn((x.size(0),x.size(1),1), device=x.device)*self.std)
            else:
                return x + dx
        else:
            if self.std > 0:
                return self.skip * x + self.gamma * self.module( self.norm(x) ) * (1+torch.randn((x.size(0),x.size(1),1), device=x.device)*self.std)
            else:
                return self.skip * x + self.gamma * self.module( self.norm(x) )

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
                dropout probability
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

    def default():
        return copy.deepcopy(Config(
            E      = None,     # размерность эмбедига
            H      = 1,        # число голов (E % H == 0 !)
            res    = 1,        # kind of skip-connections (in TransformerBlock)
            skip   = 1,        # множитель для skip петли
            causal = False,    # причинное внимание (как в GPT) иначе как в BERT
            T_max  = 2048,     # максимальное число токенов (нужно для causal==True)
            drop   = 0,        # dropout на выходе внимания
        ))

    #---------------------------------------------------------------------------

    def create(self):
        cfg = self.cfg
        assert cfg.E is not None and cfg.E > 0 and cfg.H > 0, f"must be E and H in cfg:{cfg.get_str()}"
        assert cfg.E % cfg.H == 0,  "E must be div by H: {cfg.E} % {cfg.H} != 0"

        T,E = cfg.T_max, cfg.E

        self.c_attn = nn.Linear(E,3*E)  # key, query, value projections for all heads
        self.c_proj = nn.Linear(E,  E)  # output projection

        self.att_dropout  = nn.Dropout(cfg.drop)      # regularization
        self.res_dropout  = nn.Dropout(cfg.drop)

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
        att = self.att_dropout(att)
        y = att @ v                          # (B,H,T,T) x (B,H,T,hs) -> (B,H,T,hs)
        y = y.transpose(1,2).contiguous().view(B,T,E) # re-assemble all head side by side

        # output projection
        y = self.c_proj(y)
        y = self.res_dropout(y)
        return y                               # (B,T,E)

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

    def decay(self):
        return set(self.c_attn.weight, self.c_proj.weight)
    #---------------------------------------------------------------------------

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
            drop (float = 0.0):
                dropout after fft
            res (int = 1):
                kind of skip-connections (for TransformerBlock): (1) x+f(x), (2) x+w*f(x) - w training one for all E; (3) w- training for each E
            skip (float = 1.):
                # fixed multiplier for skip loop:  skip*x + w*f(x) - can be turned off
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
        self.create()

        self.data = torch.tensor(float(0))
        self.grad = torch.tensor(float(0))

    #---------------------------------------------------------------------------

    def default():
        return copy.deepcopy(Config(
            res   = 1,         # kind of skip-connections (in TransformerBlock)
            skip  = 1.,        # fixed multiplier for skip loop
            drop  = 0,         # dropout на выходе внимания
            after = 1,         # после fft2: 1 - брать Re, 2 брать Im, 3 Re**2+Im**2
        ))

    #---------------------------------------------------------------------------

    def create(self):
        cfg = self.cfg
        self.drop  = nn.Dropout(cfg.drop)

    #---------------------------------------------------------------------------

    def forward(self, x):                      # (B,T,E)
        x = torch.fft.fft2(x)
        if self.cfg.after   == 1:
            x = x.real
        elif self.cfg.after == 2:
            x = x.imag
        else:
            x = x.real.pow(2) + x.imag.pow(2)

        x = self.drop(x)
        return x                               # (B,T,E)

    #---------------------------------------------------------------------------

    def update(self):
        pass

    #---------------------------------------------------------------------------

    def decay(self):
        return set()

    #---------------------------------------------------------------------------

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
                dropout probability in attention and mlp
            res (int=1):
                kind of skip-connections: (0) f(x) - none; (1) x+f(x), (2)  x*w+f(x) training one for all E; (3) training for each E
            causal (bool=False):
                kind of causal attention mask (True: GPT, False: BERT)
            T_max (int=2048):
                maximum number of tokens (needed for causal==True)

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
        self.cfg.set(*args)

        # мы можем одним аргументом задать параметры в att и mlp
        if 'E' in kvargs:
            self.cfg.att.E = self.cfg.mlp.input = self.cfg.mlp.output = kvargs['E']

        if 'H' in kvargs:
            self.cfg.att.H = kvargs['H']

        if 'res' in kvargs:
            self.cfg.att.res = self.cfg.mlp.res = self.cfg.fft.res = kvargs['res']

        if 'drop' in kvargs:
            self.cfg.att.drop = self.cfg.mlp.drop = self.cfg.fft.drop = kvargs['drop']

        if 'causal' in kvargs:
            self.cfg.att.causal = kvargs['causal']

        if 'T_max'  in kvargs:
            self.cfg.att.T_max = kvargs['T_max']

        if 'is_fft'  in kvargs:
            self.cfg.is_fft = kvargs['is_fft']
        if 'is_att'  in kvargs:
            self.cfg.is_att = kvargs['is_att']
        if 'is_mlp'  in kvargs:
            self.cfg.is_mlp = kvargs['is_mlp']

        self.create()

    #---------------------------------------------------------------------------

    def default():
        return copy.deepcopy(Config(
            is_fft = 0,
            is_att = 1,
            is_mlp = 1,
            att = Attention.default(),
            mlp = Config(MLP.default(), stretch=4, res=1, skip=1.0),
            fft = FFT.default(),
        ))

    #---------------------------------------------------------------------------

    def create(self):
        cfg = self.cfg
        assert cfg.att.E is not None and cfg.att.E > 0 and cfg.att.E == cfg.mlp.input and cfg.mlp.input == cfg.mlp.output,  f"TransformerBlock: embedding needs to be defined: E={cfg.E}, input={cfg.mlp.input}, output={cfg.mlp.output}"

        if cfg.is_fft:
            self.fft = Residual(FFT(cfg.fft),       dim=cfg.att.E, res=cfg.fft.res, skip=cfg.fft.skip)
        if cfg.is_att:
            self.att = Residual(Attention(cfg.att), dim=cfg.att.E, res=cfg.att.res, skip=cfg.att.skip)
        if cfg.is_mlp:
            self.mlp = Residual(MLP(cfg.mlp),       dim=cfg.att.E, res=cfg.mlp.res, skip=cfg.mlp.skip)

    #---------------------------------------------------------------------------

    def forward(self, x):
        """
        (B,T,E) -> (B,T,E)
        """
        if self.cfg.is_fft:
            x = self.fft(x)

        if self.cfg.is_att:
            x = self.att(x)

        if self.cfg.is_mlp:
            x = self.mlp(x)

        return x                                            # (B,T,E)

    #---------------------------------------------------------------------------

    def update(self):
        if self.cfg.is_fft:
            self.fft.update()

        if self.cfg.is_att:
            self.att.update()

        if self.cfg.is_mlp:
            self.mlp.update()

    #---------------------------------------------------------------------------

    def decay(self):
        res = set()
        if self.cfg.is_fft:
            res.update(self.fft.decay() )

        if self.cfg.is_att:
            res.update(self.att.decay() )

        if self.cfg.is_mlp:
            res.update(self.mlp.decay() )

        return res
    
    #---------------------------------------------------------------------------

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
                dropout in attention and mlp
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
        self.cfg.set(*args)

        # В set не передаём kvargs, чтобы не было ворнингов по E, H и т.д.
        if 'n_blocks' in kvargs:
            self.cfg.n_blocks = kvargs['n_blocks']
        if 'is_fft' in kvargs:
            self.cfg.is_fft   = kvargs['is_fft']
        if 'is_att' in kvargs:
            self.cfg.is_att   = kvargs['is_att']
        if 'is_mlp' in kvargs:
            self.cfg.is_mlp   = kvargs['is_mlp']

        # одним аргументом задаём параметры в fft, att и mlp всех блоков
        if 'E' in kvargs:
            self.cfg.block.att.E = self.cfg.block.mlp.input = self.cfg.block.mlp.output = kvargs['E']

        if 'H' in kvargs:
            self.cfg.block.att.H = kvargs['H']

        if 'res' in kvargs:
            self.cfg.block.fft.res = self.cfg.block.att.res = self.cfg.block.mlp.res = kvargs['res']

        if 'drop' in kvargs:
            self.cfg.block.fft.drop = self.cfg.block.att.drop = self.cfg.block.mlp.drop = kvargs['drop']

        if 'after' in kvargs:
            self.cfg.block.fft.after = kvargs['after']

        if 'causal' in kvargs:
            self.cfg.block.att.causal = kvargs['causal']

        if 'T_max'  in kvargs:
            self.cfg.block.att.T_max = kvargs['T_max']

        self.create()

    #---------------------------------------------------------------------------

    def default():
        return copy.deepcopy(Config(
            n_blocks  = 1,     # число слоёв трансформера
            is_fft    = 0,     # there is a FFT (FNet) block, can be a list (for each block)
            is_att    = 1,     # there is an attention block, can be a list (for each block)
            is_mlp    = 1,     # there is an MLP block, can be a list (for each block)
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
            block = TransformerBlock(self.cfg.block, is_fft=is_fft[i], is_att=is_att[i], is_mlp=is_mlp[i])
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

    def plot(self, w=12, h=3, eps=1e-8):
        idx = np.arange(len(self.blocks))

        fig, ax = plt.subplots(1,1, figsize=(w, h))
        
        plt.text(0,0,f" mult\n std\n", ha='left', transform = ax.transAxes, fontsize=8)
        
        ax.grid(ls=":")
        ax.set_xticks(idx)

        ax.bar(idx, idx*2)
        ax.bar(idx, idx*3, bottom = idx*2)

        plt.show()

    #---------------------------------------------------------------------------

    def unit_test():
        B, T, E, H = 1, 10, 128, 16
        m = Transformer(E=E, H=H, n_blocks=5, is_fft=[1,1,0,0,0], is_att=[0,0,1,1,1])
        x = torch.rand(B,T,E)  # (B,T,E) = (batch, token, embedding)
        y = m(x)               # (B,T,E) -> (B,T,E)
        m.plot()

        #for block in m.blocks: print(block.cfg)

        res = x.shape == y.shape
        if not res:
            print(f"!! Transformer: x.shape={x.shape} != y.shape={y.shape}")
        else:
            print(f"ok Transformer: {tuple(x.shape)} -> {tuple(y.shape)}")
        return res

#===============================================================================
