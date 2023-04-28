import math, copy
import torch, torch.nn as nn

from ..utils   import Config
from .mlp      import MLP

#===============================================================================

class SelfAttention(nn.Module):
    """
    base on: https://github.com/karpathy/nanoGPT/blob/master/model.py
    """
    def __init__(self,  *args, **kvargs) -> None:
        """
        Self Attention
        Args:         
            * `E:int`  - tokens embedding dim
            * `H:int`  - number of heads E % H == 0 !            
            * `drop=0` - dropout in attention
            * `res=1` - kind of skip-connections (0: none, 1: usial, 2: train one for all E, 3: train for each E)
            * `casual = False` - kind of casual attention mask (True: GPT, False: Bert)
            * `T_max  = 2048` -  maximum number of tokens (needed for causal==True)
        """
        super().__init__()
        self.cfg = SelfAttention.default()
        self.cfg.set(*args, **kvargs)
        self.create()

    def default():
        return copy.deepcopy(Config(
            E  = None,          # размерность эмбедига
            H  = 1,             # число голов (E % H == 0 !)
            res= 1,             # kind of skip-connections
            causal = False,     # причинное внимание (как в GPT) иначе как в BERT
            T_max = 2048,       # максимальное число токенов (нужно для causal==True)
            drop   = 0,         # dropout на выходе внимания
        ))
    
    def create(self):
        cfg = self.cfg
        assert cfg.E > 0 and cfg.H > 0, f"must be E and H in cfg:{cfg.get()}"
        assert cfg.E % cfg.H == 0,  "E must be div by H: {cfg.E} % {cfg.H} != 0"
        
        T,E = cfg.T_max, cfg.E

        self.c_attn = nn.Linear(E,3*E)  # key, query, value projections for all heads
        self.c_proj = nn.Linear(E,  E)  # output projection

        self.att_dropout  = nn.Dropout(cfg.drop)      # regularization
        self.res_dropout  = nn.Dropout(cfg.drop)

        if cfg.causal:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(T, T)).view(1,1,T,T))

    def forward(self, x):  # (B,T,E)
        B,T,E = x.size() # batch size, sequence length, embedding dimensionality (E)
        assert E == self.cfg.E, f"wrong input {E} != {self.cfg.E}"

        # calculate query, key, values for all heads and move head forward to be the batch dim
        q,k,v  = self.c_attn(x).split(self.cfg.E, dim=2)
        k = k.view(B,T, self.cfg.H, E // self.cfg.H).transpose(1,2) # (B, nh, T, hs) hs = E/nh
        q = q.view(B,T, self.cfg.H, E // self.cfg.H).transpose(1,2) # (B, nh, T, hs)
        v = v.view(B,T, self.cfg.H, E // self.cfg.H).transpose(1,2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs)x(B, nh, hs, T) -> (B, nh, T,T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.cfg.causal:  # заполняем верхний угол треугольной матрицы -inf
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.att_dropout(att)
        y = att @ v                          # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B,T,E) # re-assemble all head side by side

        # output projection
        y = self.c_proj(y)
        y = self.res_dropout(y)        
        return y                               # (B,T,E)

#===============================================================================

class FFT(nn.Module):
    """
    see: "FNet: Mixing Tokens with Fourier Transforms" https://arxiv.org/abs/2105.03824
    """
    def __init__(self,  *args, **kvargs) -> None:
        """
        FFT Block from FNet
        Args:         
            * `E:int`  - tokens embedding dim
            * `drop=0` - dropout after attention
            * `res=1` - kind of skip-connections (0: none, 1: usial, 2: train one for all E, 3: train for each E)
            * `get = 1` - after fft2: (1) take Re, (2) take Im, (3) Re**2+Im**2 
        """
        super().__init__()
        self.cfg = FFT.default()
        self.cfg.set(*args, **kvargs)
        self.create()

    def default():
        return copy.deepcopy(Config(            
            res  = 1,       # kind of skip-connections
            drop = 0,       # dropout на выходе внимания
            get  = 1,       # после fft2: 1 - брать Re, 2 брать Im, 3 Re**2+Im**2 
        ))
    
    def create(self):
        cfg = self.cfg        
        self.drop  = nn.Dropout(cfg.drop)            

    def forward(self, x):                      # (B,T,E)
        x = torch.fft.fft2(x)
        if self.cfg.get   == 1:
            x = x.real
        elif self.cfg.get == 2:
            x = x.imag
        else:
            x = x.real.pow(2) + x.imag.pow(2)

        x = self.drop(x)
        return x                               # (B,T,E)

#===============================================================================


class  TransformerBlock(nn.Module):
    def __init__(self,  *args, **kvargs) -> None:
        """
        One Transformer Block (it is all you need)
        Args:         
            * `E:int`  -  tokens embedding dim
            * `H:int`  - number of heads E % H == 0 !            
            * `drop=0` - dropout in attention and mlp
            * `res=1`  - kind of skip-connections (0: none, 1: usial, 2: train one for all E, 3: train for each E)
            * `casual = False` - kind of casual attention mask (True: GPT, False: Bert)
            * 'T_max = 2048'- maximum number of tokens (needed for causal==True)
        """
        super().__init__()
        self.cfg = TransformerBlock.default()
        self.cfg.set(*args)

        # мы можем одним аргументом задать параметры в att и mlp
        if 'E' in kvargs:            
            self.cfg.att.E      = kvargs['E']
            self.cfg.mlp.input  = kvargs['E']
            self.cfg.mlp.output = kvargs['E']

        if 'H' in kvargs:
            self.cfg.att.H = kvargs['H']

        if 'res' in kvargs:            
            self.cfg.att.res = kvargs['res']
            self.cfg.mlp.res = kvargs['res']
            self.cfg.fft.res = kvargs['res']

        if 'drop' in kvargs:
            self.cfg.att.drop = kvargs['drop']
            self.cfg.mlp.drop = kvargs['drop']
            self.cfg.fft.drop = kvargs['res']

        if 'casual' in kvargs:
            self.cfg.att.casual = kvargs['casual']

        if 'T_max'  in kvargs:
            self.cfg.att.T_max = kvargs['T_max']

        self.create()

    def default():
        return copy.deepcopy(Config(               
            is_fft = 0,
            is_att = 1,
            is_mlp = 1,                     
            att = SelfAttention.default(),
            mlp = Config(MLP.default(), stretch=4, res=1),
            fft = FFT.default(),
        ))

    def create(self): 
        cfg = self.cfg
        assert cfg.att.E > 0 and cfg.att.E == cfg.mlp.input and cfg.mlp.input == cfg.mlp.output,  f"TransformerBlock: embedding needs to be defined: E={cfg.E}, input={cfg.mlp.input}, output={cfg.mlp.output}"


        if cfg.is_mlp:
            self.ln_2 = nn.LayerNorm(cfg.att.E)
            self.mlp  = MLP(cfg.mlp)
    
        if cfg.is_att:
            self.ln_1 = nn.LayerNorm (cfg.att.E)
            self.att  = SelfAttention(cfg.att)

            cfg.att.res = max(0, min(3, cfg.att.res))
            if   cfg.att.res == 3:     # train multiplier for each components
                self.w_att = nn.Parameter( torch.ones( cfg.att.E ) )            
            elif cfg.att.res == 2:     # train common multiplier
                self.w_att   = nn.Parameter( torch.ones(1) )            
            else:                     # constant multiplayer (0 or 1)           
                self.register_buffer("w_att", torch.tensor(float(cfg.att.res)))                        

            cfg.mlp.res = max(0, min(3, cfg.mlp.res))
            if   cfg.mlp.res == 3:     # train multiplier for each components            
                self.w_mlp = nn.Parameter( torch.ones( cfg.att.E ) )
            elif cfg.mlp.res == 2:     # train common multiplier            
                self.w_mlp   = nn.Parameter( torch.ones(1) )
            else:                     # constant multiplayer   (0 or 1)                     
                self.register_buffer("w_mlp", torch.tensor(float(cfg.mlp.res)))            

        if cfg.is_fft:
            self.ln_0 = nn.LayerNorm (cfg.att.E)
            self.fft = FFT(cfg.fft)

    def forward(self, x):                          # (B,T,E)
        """
        Классический highway: w=sigmoid(linear(x)); x*w + (1-w)*mlp, ...
        не работал для нейтрино. Возможно это негативный эффект постоянного
        умножения x (и потом градиента) на число меньшее 1:  x*0.5*0.5*...
        Не работал также linear(x) + mlp, с начальной единичной матрицей :(
        
        Карпатов сделал ln на входе, хотя многие рисуют на выходе.
        Это сказывается тоько на первом и последнем блоке. 
        Наверное как у Карпатова правильнее (нормируем перед вычислениями).
        """
        if self.cfg.is_fft:
            x = x * self.w_fft + self.fft(self.ln_0(x))
        if self.cfg.is_att:
            x = x * self.w_att + self.att(self.ln_1(x))
        if self.cfg.is_mlp:
            x = x * self.w_mlp + self.mlp(self.ln_2(x))
        return x                                  # (B,T,E)


#===============================================================================

class  Transformer(nn.Module):
    """
        Args:         
            * `E:int` - tokens embedding dim
            * `H` - number of heads E % H == 0 !
            * `n_blocks=1` - number of transformer blocks
            * `drop=0` - dropout in attention and mlp
            * `res=1` - kind of skip-connections (0: none, 1: usial, 2: train one for all E, 3: train for each E)
            * `casual=False` - kind of casual attention mask (True: GPT, False: Bert)
            * 'T_max = 2048'- maximum number of tokens (needed for causal==True)
            * is_fft = 0,    # there is a FFT (FNet) block, can be a list (for each block)
            * is_att = 1,    # there is an attention block, can be a list (for each block)
            * is_mlp = 1,    # there is an MLP block, can be a list (for each block)
    """    
    def __init__(self,  *args, **kvargs) -> None:
        """
        Transformer is all you need

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
                kind of skip-connections (0: none, 1: usial, 2: train one for all E, 3: train for each E)
            casual (bool=False):  
                kind of casual attention mask (True: GPT, False: Bert)
            T_max  (int=2048): 
                maximum number of tokens (needed for causal==True)
            is_fft  (int=0):   
                there is a FFT (FNet) block, can be a list (for each block)
            is_att  (int=0):   
                there is an attention block, can be a list (for each block)
            is_mlp  (int=0):   
                there is an MLP block, can be a list (for each block)

        Example:
        ```
        m = Transformer(E=64, H=8, n_blocks=3, is_fft=[1,1,0], is_att=[0,0,1])

        x = torch.rand(1,10,64)  # (B,T,E) = (batch, token, embedding)
        y = m(x)                 # (B,T,E) -> (B,T,E)
        ```                
        """
        super().__init__()        
        self.cfg = Transformer.default()
        self.cfg.set(*args)

        if 'n_blocks' in kvargs:
            self.cfg.n_blocks = kvargs['n_blocks']        

        # мы можем одним аргументом задать параметры в att и mlp всех блоков
        if 'E' in kvargs:        
            self.cfg.block.att.E      = kvargs['E']
            self.cfg.block.mlp.input  = kvargs['E']
            self.cfg.block.mlp.output = kvargs['E']

        if 'H' in kvargs:
            self.cfg.block.att.H = kvargs['H']

        if 'res' in kvargs:            
            self.cfg.block.att.res = kvargs['res']
            self.cfg.block.mlp.res = kvargs['res']

        if 'drop' in kvargs:
            self.cfg.block.att.drop = kvargs['drop']
            self.cfg.block.mlp.drop = kvargs['drop']

        if 'casual' in kvargs:
            self.cfg.block.att.casual = kvargs['casual']

        if 'T_max'  in kvargs:
            self.cfg.block.att.T_max = kvargs['T_max']

        self.create()

    def default():
        return copy.deepcopy(Config(
            n_blocks  = 1, # число слоёв трансформера
            is_fft = 0,    # there is a FFT (FNet) block, can be a list (for each block)
            is_att = 1,    # there is an attention block, can be a list (for each block)
            is_mlp = 1,    # there is an MLP block, can be a list (for each block)
            block = TransformerBlock.default()            
        ))

    def create(self):        
        self.blocks= nn.ModuleList([TransformerBlock(self.cfg.block) for _ in range(self.cfg.n_blocks)])

    def forward(self, x):                          # (B,T,E)
        for block in self.blocks:            
            x = block(x)                           # (B,T,E)
        return x

#===============================================================================
