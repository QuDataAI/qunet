import math, copy
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
    def __init__(self, module, dim, res = 1, skip=1., gamma=0.1):
        super().__init__()
        self.module  = module
        if   res == 3:                # training multiplier for each components
            self.gamma = nn.Parameter( torch.empty( dim ).fill_(float(gamma)) )
        elif res == 2:                # training common multiplier
            self.gamma = nn.Parameter( torch.tensor(float(gamma)) )     # 0 ??? Julius Ruseckas
        else:                         # constant multiplayer
            self.register_buffer("gamma", torch.tensor(1.) )

        self.register_buffer("skip", torch.tensor(float(skip)) )

        self.norm  = nn.LayerNorm (dim)

        self.debug  = False         # see forward
        self.beta   = 0.9           # value of smoothing
        self.sqr_x  = None          # average of square value of block input
        self.sqr_dx = None
        self.std    = torch.tensor(float(0))

        self.gamma_d   = None       # average for gamma data and grad
        self.gamma_g   = None       # (see update)

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
            gamma  = 0.1,
            att = Attention.default(),
            mlp = Config(MLP.default(), stretch=4, res=1, skip=1.0),
            fft = FFT.default(),
        ))

    #---------------------------------------------------------------------------

    def create(self):
        cfg = self.cfg
        assert cfg.att.E is not None and cfg.att.E > 0 and cfg.att.E == cfg.mlp.input and cfg.mlp.input == cfg.mlp.output,  f"TransformerBlock: embedding needs to be defined: E={cfg.E}, input={cfg.mlp.input}, output={cfg.mlp.output}"

        self.fft = self.att = self.mlp = None

        if cfg.is_fft:
            self.fft = Residual(FFT(cfg.fft),       dim=cfg.att.E, res=cfg.fft.res, skip=cfg.fft.skip, gamma=cfg.gamma)
        if cfg.is_att:
            self.att = Residual(Attention(cfg.att), dim=cfg.att.E, res=cfg.att.res, skip=cfg.att.skip, gamma=cfg.gamma)
        if cfg.is_mlp:
            self.mlp = Residual(MLP(cfg.mlp),       dim=cfg.att.E, res=cfg.mlp.res, skip=cfg.mlp.skip, gamma=cfg.gamma)

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

    def debug(self, value=True, beta=None):
        if self.fft is not None:
            self.fft.debug = value
            if beta is not None:
                self.fft.beta = beta

        if self.att is not None:
            self.att.debug = value
            if beta is not None:
                self.att.beta = beta

        if self.mlp is not None:
            self.mlp.debug = value
            if beta is not None:
                self.mlp.beta = beta

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

        if 'gamma' in kvargs:
            self.cfg.block.gamma =  kvargs['gamma']
        else:
            self.cfg.block.gamma = self.cfg.gamma

        # одним аргументом задаём параметры в fft, att и mlp всех блоков
        E = kvargs['E'] if 'E' in kvargs else (self.cfg.E if self.cfg.has('E') else self.cfg.block.att.E)
        self.cfg.block.att.E = self.cfg.block.mlp.input = self.cfg.block.mlp.output = E

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

    def default():
        return copy.deepcopy(Config(
            n_blocks  = 1,     # число слоёв трансформера
            is_fft    = 0,     # there is a FFT (FNet) block, can be a list (for each block)
            is_att    = 1,     # there is an attention block, can be a list (for each block)
            is_mlp    = 1,     # there is an MLP block, can be a list (for each block)
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

    def debug(self, value=True, beta=None):
        for block in self.blocks:
            block.debug(value, beta)

    #---------------------------------------------------------------------------

    def plot(self, w=12, h=3, eps=1e-8, bar_width = 0.25, info=False):
        
        idx = np.arange(len(self.blocks))

        fig, ax = plt.subplots(1,1, figsize=(w, h))

        ax.grid(ls=":")
        ax.set_xticks(idx)
        ax.set_yscale('log')
        ax.set_ylabel("dx/x");  ax.set_xlabel("blocks");

        plt.text(0,0,f" skip\n std\n", ha='left', transform = ax.transAxes, fontsize=6, family="monospace")
                
        fft_dv = [ (b.fft.sqr_dx / (b.fft.sqr_x+eps)).sqrt().cpu().item() if b.fft is not None  else 0  for b in self.blocks]
        att_dv = [ (b.att.sqr_dx / (b.att.sqr_x+eps)).sqrt().cpu().item() if b.att is not None  else 0  for b in self.blocks]
        mlp_dv = [ (b.mlp.sqr_dx / (b.mlp.sqr_x+eps)).sqrt().cpu().item() if b.mlp is not None  else 0  for b in self.blocks]
        #ax.set_ylim(ymin=0, ymax=max(np.max(fft), np.max(att), np.max(mlp)*1.1) ) 
        
        ax.bar(idx,              fft_dv, width = bar_width, edgecolor ='grey', alpha=0.5)
        ax.bar(idx +  bar_width, att_dv, width = bar_width, edgecolor ='grey', alpha=0.5)
        ax.bar(idx +2*bar_width, mlp_dv, width = bar_width, edgecolor ='grey', alpha=0.5)

        ymin, ymax = ax.get_ylim()
        for i,b in enumerate(self.blocks):
            if b.fft is not None:
                plt.text(i,            ymin, f"{b.fft.skip.cpu().item():.1f}\n{b.fft.std.cpu().item():.1f}\nfft", ha='center', fontsize=6, family="monospace")
            if b.att is not None:
                plt.text(i+bar_width,  ymin, f"{b.att.skip.cpu().item():.1f}\n{b.att.std.cpu().item():.1f}\natt", ha='center', fontsize=6, family="monospace")
            if b.mlp is not None:
                plt.text(i+2*bar_width,ymin, f"{b.mlp.skip.cpu().item():.1f}\n{b.mlp.std.cpu().item():.1f}\nmlp", ha='center', fontsize=6, family="monospace")

        ax2 = ax.twinx()
        fft_gamma_d = [ (b.fft.gamma_d).sqrt().cpu().item() if b.fft is not None and b.fft.gamma_d is not None else 0  for b in self.blocks]
        att_gamma_d = [ (b.att.gamma_d).sqrt().cpu().item() if b.att is not None and b.att.gamma_d is not None else 0  for b in self.blocks]
        mlp_gamma_d = [ (b.mlp.gamma_d).sqrt().cpu().item() if b.mlp is not None and b.mlp.gamma_d is not None else 0  for b in self.blocks]
        ax2.plot(idx,               fft_gamma_d, marker=".")
        ax2.plot(idx +   bar_width, att_gamma_d, marker=".")
        ax2.plot(idx + 2*bar_width, mlp_gamma_d, marker=".")
        ax2.set_ylabel("gamma")

        ax3 = ax.twinx()
        ax3.set_yscale('log')
        fft_gamma_g = [ (b.fft.gamma_g).sqrt().cpu().item() if b.fft is not None and b.fft.gamma_g is not None else 0  for b in self.blocks]
        att_gamma_g = [ (b.att.gamma_g).sqrt().cpu().item() if b.att is not None and b.att.gamma_g is not None else 0  for b in self.blocks]
        mlp_gamma_g = [ (b.mlp.gamma_g).sqrt().cpu().item() if b.mlp is not None and b.mlp.gamma_g is not None else 0  for b in self.blocks]
        ax3.plot(idx,               fft_gamma_g, ":", marker=".")
        ax3.plot(idx +   bar_width, att_gamma_g, ":", marker=".")
        ax3.plot(idx + 2*bar_width, mlp_gamma_g, ":", marker=".")        
        ax3.set_ylabel("gamma grad")
        ax3.spines["right"].set_position(("outward", 50))

        plt.show()
        if info:
            return {
                'fft_dv': fft_dv, 'fft_gamma_d': fft_gamma_d, 'fft_gamma_g': fft_gamma_g,
                'att_dv': att_dv, 'att_gamma_d': att_gamma_d, 'att_gamma_g': att_gamma_g,
                'mlp_dv': mlp_dv, 'mlp_gamma_d': mlp_gamma_d, 'mlp_gamma_g': mlp_gamma_g,
            }
        
    #---------------------------------------------------------------------------

    def unit_test():
        B, T, E, H = 1, 10, 128, 16
        cfg = Config(E=E, H=H, n_blocks=10, is_fft=1, is_att=1, res=2, gamma=0.2)
        m = Transformer(cfg)
        x = torch.rand(B,T,E)  # (B,T,E) = (batch, token, embedding)

        m.debug(True)        
        y = m(x)               # (B,T,E) -> (B,T,E)
        y.mean().backward()
        m.update()
        print( m.plot(info=True) )

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

#===============================================================================
