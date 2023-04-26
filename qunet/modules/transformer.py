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
            * `max_T  = 2048` -  maximum number of tokens (needed for causal==True)
        """
        super().__init__()
        self.cfg = SelfAttention.default()
        self.cfg.set(*args, **kvargs)
        self.create()

    def default():
        return copy.deepcopy(Config(
            E  = None,          # размерность эмбедига
            H  = 1,             # число голов (E % H == 0 !)
            causal = False,     # причинное внимание (как в GPT) иначе как в BERT
            max_T = 2048,       # максимальное число токенов (нужно для causal==True)
            drop   = 0,         # dropout на выходе внимания
        ))
    
    def create(self):
        cfg = self.cfg
        assert cfg.E > 0 and cfg.H > 0, f"must be E and H in cfg:{cfg.get()}"
        assert cfg.E % cfg.H == 0,  "E must be div by H: {cfg.E} % {cfg.H} != 0"

        cfg = self.cfg
        T,E = cfg.max_T, cfg.E

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

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
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
        y = y.transpose(1, 2).contiguous().view(B,T,E) # re-assemble all head outputs side by side

        # output projection
        y = self.res_dropout(self.c_proj(y))
        return y                               # (B,T,E)

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
        if 'drop' in kvargs:
            self.cfg.att.drop = kvargs['att']
            self.cfg.mlp.drop = kvargs['drop']
        if 'casual' in kvargs:
            self.cfg.att.casual = kvargs['casual']

        self.create()

    def default():
        return copy.deepcopy(Config(            
            att = SelfAttention.default(),
            mlp = MLP.default(),
        ))

    def create(self): 
        cfg = self.cfg
        assert cfg.att.E > 0 and cfg.att.E == cfg.mlp.input and cfg.mlp.input == cfg.mlp.output,  f"TransformerBlock: embedding needs to be defined: E={cfg.E}, input={cfg.mlp.input}, output={cfg.mlp.output}"

        self.ln_1 = nn.LayerNorm (cfg.att.E)
        self.att  = SelfAttention(cfg.att)

        self.mlp = None
        if cfg.mlp.stretch > 0:
            self.ln_2 = nn.LayerNorm(cfg.att.E)
            self.mlp  = MLP(cfg.mlp)

        if   cfg.att.res == 3:     # train multiplier for each components
            self.w_att = nn.Parameter( torch.ones( cfg.att.E ) )            
        elif cfg.att.res == 2:     # train common multiplier
            self.w_att   = nn.Parameter( torch.ones(1) )            
        else:                     # constant multiplayer
            self.register_buffer("w_att", torch.Tensor(cfg.att.res))            

        if   cfg.mlp.res == 3:     # train multiplier for each components            
            self.w_mlp = nn.Parameter( torch.ones( cfg.att.E ) )
        elif cfg.mlp.res == 2:     # train common multiplier            
            self.w_mlp   = nn.Parameter( torch.ones(1) )
        else:                     # constant multiplayer            
            self.register_buffer("w_mlp", torch.Tensor(cfg.mlp.res))


    def forward(self, x):                          # (B,T,E)
        """
        Классический highway: w=sigmoid(linear(x)); x*w + (1-w)*mlp, ...
        не работал для нейтрино. Возможно это негативный эффект постоянного
        умножения x (и потом градиента) на число меньшее 1:  x*0.5*0.5*...
        Не работал также linear(x) + mlp, с наяальной единичной матрицей :(
        """
        x = x * self.w_att + self.att(self.ln_1(x))
        x = x * self.w_mlp + self.mlp(self.ln_2(x))
        return x                                  # (B,T,E)


#===============================================================================

class  Transformer(nn.Module):
    def __init__(self,  *args, **kvargs) -> None:
        """
        Transformer is all you need
        Args:         
            * `E:int` - tokens embedding dim
            * `H` - number of heads E % H == 0 !
            * `n_blocks=1` - number of transformer blocks
            * `drop=0` - dropout in attention and mlp
            * `res=1` - kind of skip-connections (0: none, 1: usial, 2: train one for all E, 3: train for each E)
            * `casual=False` - kind of casual attention mask (True: GPT, False: Bert)
        """
        super().__init__()        
        self.cfg = Transformer.default()
        self.cfg.set(*args)

        if 'n_blocks' in kvargs:
            self.cfg.n_blocks = kvargs['n_blocks']

        # мы можем одним аргументом задать параметры в att и mlp всех блоков
        if 'E' in kvargs:        
            self.cfg.block.att.E    = kvargs['E']
            self.cfg.block.mlp.input  = kvargs['E']
            self.cfg.block.mlp.output = kvargs['E']
        if 'H' in kvargs:
            self.cfg.block.att.H = kvargs['H']
        if 'res' in kvargs:            
            self.cfg.block.att.res = kvargs['res']
            self.cfg.block.mlp.res = kvargs['res']
        if 'drop' in kvargs:
            self.cfg.block.att.drop = kvargs['att']
            self.cfg.block.mlp.drop = kvargs['drop']
        if 'casual' in kvargs:
            self.cfg.block.att.casual = kvargs['casual']

        self.create()

    def default():
        return copy.deepcopy(Config(
            n_blocks  = 1,             # число слоёв трансформера
            block = Config(                       
                att = SelfAttention.default(),
                mlp = MLP.default(),
            ),
        ))

    def create(self):        
        self.blocks= nn.ModuleList([TransformerBlock(self.cfg.block) for _ in range(self.cfg.n_blocks)])

    def forward(self, x):                          # (B,T,E)
        for i, block in enumerate(self.blocks):
            #if CFG.frozen and self.training and i > CFG.L_frozen: torch.set_grad_enabled(True)
            x = block(x)                           # (B,T,E)
        return x

#===============================================================================
