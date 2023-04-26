"""
Архитектуры:
    * MLP               - полносвязная сеть с одним скрытм слоем  (B,*,n)  -> (B,*,m)
    * CNN               - свёрточная сеть                         (B,C,H,W)-> (B,C',H',W')
    * SelfAttention     - самовнимание (причинное или нет)        (B,T,E)  -> (B,T,E)
    * TransformerBlock  - блок трансформера                       (B,T,E)  -> (B,T,E)
    * PointsBlock       - блок повортора координат                (B,T,E)  -> (B,T,E)

Обучение: (см. nnet_trainer)
    * Data
    * Trainer

Полезно:
    * nn.Bilinear       -  bilinear transformation:  x1 A x2 + b
    * torchview         - https://github.com/mert-kurttutan/torchview

См. примеры в конце файла в функции main
                                                            (c) 2023 - QuData.com (steps)
"""
import os, math, copy, time, datetime
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn
from   tqdm.auto import tqdm

from .utils   import Config

#========================================================================================

class MLP(nn.Module):
    def __init__(self,  *args, **kvargs) -> None:
        """
        Fully connected network with one or more hidden layers: (B,*, input) -> (B,*, output).

        Args:
            * `input  :int`         - number of inputs > 0
            * `output :int`         - number of outputs > 0
            * `hidden :int or list` - number of neurons in the hidden layer
            * `stretch = 4`         - if there is, then hidden = int(stretch*input)
            * `fun = 'gelu'`        - activation function: gelu, relu, sigmoid, tanh
            * `drop=  0`            - dropout at the output of the hidden layer

        If there is more than one layer - cfg['hidden'] is a list with a list of the number of neurons in each layer
        There may be no hidden layer: hidden == 0 or == [] or stretch == 0,
        then it's a normal input -> output line layer with no activation function
            
        Example:
        ```
            mlp = MLP(input=32, stretch=4, output=1)
            y = mlp( torch.randn(1, 32) )
        ```
        Can be created from config:
        ```
            cfg = MLP.default()         
            cfg(input = 3, output = 1)  
            mlp = MLP(cfg)              
        ```
        And also from the config and arguments:
        ```
            mlp = MLP(cfg, hidden=[128, 512])
        ```
        """
        super().__init__()
        self.cfg = MLP.default()
        self.cfg.set(*args, **kvargs)
        self.create()

    def default():
        return copy.deepcopy(Config(
            input   = None,           # number of inputs > 0
            output  = None,           # number of outputs > 0
            hidden  = None,           # number of neurons in the hidden layer (int or list)
            stretch = 4,              # if there is, then hidden = int(stretch*input)
            fun     = 'gelu',         # activation function: gelu, relu, sigmoid, tanh
            drop    =  0,             # dropout at the output of the hidden layer
        ))

    def forward(self, x):
        x = self.layers(x)
        return x

    def prepare(self):
        cfg=self.cfg
        assert cfg.input is not None  and cfg.output is not None,  f'MLP: wrong input/output: {cfg.get()}'

        if type(cfg.hidden) is list:
            self.neurons = [cfg.input] + cfg.hidden + [cfg.output]
        else:
            if (cfg.hidden is None) and (cfg.stretch is not None) and cfg.stretch > 0:
                cfg.hidden = int(cfg.stretch * cfg.input)

            if cfg.hidden is None or cfg.hidden <= 0:
                self.neurons = [cfg.input, cfg.output]                
            else:
                self.neurons = [cfg.input, cfg.hidden, cfg.output]

        if cfg.fun not in ['gelu', 'relu', 'sigmoid', 'tanh']:
            print(f"MLP warning: unknown activation function {cfg.fun}, set to gelu")
            cfg.fun  = 'gelu'

    def create(self):
        self.prepare()
        seq = []
        for i in range (1, len(self.neurons)):
            seq += [ nn.Linear(self.neurons[i-1],  self.neurons[i]) ]
            if i+1 < len(self.neurons):
                if   self.cfg.fun == 'gelu':    seq += [ nn.GELU() ]
                elif self.cfg.fun == 'relu':    seq += [ nn.ReLU() ]
                elif self.cfg.fun == 'sigmoid': seq += [ nn.Sigmoid() ]
                elif self.cfg.fun == 'tanh':    seq += [ nn.Tanh() ]                
                seq += [ nn.Dropout(self.cfg.drop) ]
        self.layers = nn.Sequential(*seq)

#========================================================================================

class CNN(nn.Module):
    def __init__(self,  *args, **kvargs) -> None:
        """
        Simple convolutional network: (B,C,H,W) ->  (B,C',H',W')

        The number of layers is set by the channel parameter. This is the number of channels at the output of each layer.
        For example input=(3,32,32) and channel=[8,16] will create two CNN layers and the output of the module will be 16 channels.
        The channel output size (C',H',W') is in cfg.output after the module is created.
        The remaining parameters are specified either as lists (for each layer) or as numbers (then they will be the same in each layer).

        Args:
            * `input= None`:  input tensor shape:: (channels, height, width)            
            * `channel:list`:  number of channels in each layer
            * `kernel  = 3`:   int or list: size of the convolutional kernel
            * `stride   = 1`:  int or list: stride of the convolutional kernel
            * `padding  = 1`:  int or list: padding around the image
            * `pool_ker = 2`:  int or list: max-pooling kernel
            * `pool_str = 2`:  int or list: stride of max-pooling kernel
            * `drop     = 0`:  int or list: dropout after each layer
        
        Example:
        ```
            cnn = CNN(input=(3,32,32), channel=[16,32], kernel=3, pool_ker=[0,2])
        ```
        """
        super().__init__()
        self.cfg = CNN.default()
        self.cfg.set(*args, **kvargs)
        self.create()

    def default():
        return copy.deepcopy(Config(
            input    = None,       # input tensor shape:: (channels, height, width)
            output   = None,       # output tensor shape;  sets in create()
            channel  = None,       # number of channels in each layer
            kernel   = 3,          # int or list: size of the convolutional kernel
            stride   = 1,          # int or list: stride of the convolutional kernel
            padding  = 1,          # int or list: padding around the image
            pool_ker = 2,          # int or list: max-pooling kernel
            pool_str = 2,          # int or list: stride of max-pooling kernel
            drop     = 0,          # int or list: dropout after each layer
        ))

    def forward(self, x):
        x = self.layers(x)
        return x

    def prepare(self):        
        cfg = self.cfg
        assert type(cfg.input) == tuple, "CNN: You must define input shape of image (C,H,W)"
        if type(cfg.channel) == int:
            cfg.channel = [cfg.channel]
        assert type(cfg.channel) == list, f"CNN: You must define channel (int or list of int): {cfg.channel}"

        n = len(cfg.channel)
        if type(cfg.drop)     == int:   cfg.drop     = [cfg.drop]    * n
        if type(cfg.kernel)   == int:   cfg.kernel   = [cfg.kernel]  * n
        if type(cfg.stride)   == int:   cfg.stride   = [cfg.stride]  * n
        if type(cfg.padding)  == int:   cfg.padding  = [cfg.padding] * n
        if type(cfg.pool_ker) == int:   cfg.pool_ker = [cfg.pool_ker]* n
        if type(cfg.pool_str) == int:   cfg.pool_str = [cfg.pool_str]* n
        if type(cfg.drop)     == float or type(cfg.drop)==int: cfg.drop=[cfg.drop]*n        

    def create(self):
        self.prepare()
        c, w, h  =  self.cfg.input
        channels = [ c ] + self.cfg.channel
        layers = []
        for i in range(len(channels)-1):
            kernel, stride     = self.cfg.kernel  [i], self.cfg.stride[i]
            padding            = self.cfg.padding [i]
            pool_ker, pool_str = self.cfg.pool_ker[i], self.cfg.pool_str[i]

            layers +=  [
                nn.Conv2d(channels[i],channels[i+1], kernel_size=kernel, stride=stride, padding=padding),
                nn.ReLU()]
            h = int( (h + 2*padding - kernel) / stride + 1)
            w = int( (w + 2*padding - kernel) / stride + 1)

            if pool_ker > 1:
                layers += [ nn.MaxPool2d(kernel_size=pool_ker, stride=pool_str) ]
                h = int( (h - pool_ker) / pool_str + 1)
                w = int( (w - pool_ker) / pool_str + 1)
            
            layers += [ nn.Dropout(p=self.cfg.drop[i]) ]

        self.cfg.output =  (channels[-1], w, h)
        self.layers =  nn.Sequential(*layers)

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

class  PointsBlock(nn.Module):
    def __init__(self,  *args, **kvargs) -> None:
        super().__init__()
        self.cfg = PointsBlock.default()
        self.cfg.set(*args, **kvargs)
        self.create()

    def default():
        return copy.deepcopy(Config(
            E  = 64,          # размерность эмбедига
            max  = False,
            mean = True,
            res  = 1,           # residual петли (0-нет, 1-обычные, 2-тренеруемые)
            mlp1 = MLP.default(),
            mlp2 = MLP.default(),
        ))

    def create(self):
        cfg = self.cfg
        assert cfg.mean or cfg.max, f"PointsBlock need mean or/and max, cfg={cfg.get()}"
        self.cfg.mlp1(input = cfg.E, output = cfg.E)
        self.cfg.mlp2(input = cfg.E, output = cfg.E)

        E, E2 = cfg.E, cfg.E
        self.ln_1 = nn.LayerNorm(E)
        self.ln_2 = nn.LayerNorm(E)

        n_cat = 2 if cfg.max and cfg.mean else 1
        self.mlp_1 = MLP(input=E, stretch=cfg.mlp1.stretch, output=E2)
        self.fc_w  = nn.Linear(n_cat*E2, E*E)
        self.fc_b  = nn.Linear(n_cat*E2, E)
        self.mlp_2 = MLP(input=E, stretch=cfg.mlp1.stretch, output=E)

        if cfg.res == 2:
            self.w_1   = nn.Parameter( torch.ones(1) )
            self.w_2   = nn.Parameter( torch.ones(1) )
        else:
            self.w_1   = cfg.res
            self.w_2   = cfg.res

        self.E       = E

    def forward(self, x):                                       # (B,T, E)
        x = self.ln_1(x)                                        # (B,T, E)
        y = self.mlp_1(x)                                       # (B,T, E')
        agg = []
        if self.cfg.mean:
            agg.append( y.mean(dim=1) )
        if self.cfg.max:
            agg.append( y.max(dim=1)[0] )
        y = torch.cat(agg, dim=1)                               # (B, n_cat*E')
        w = self.fc_w(y).view(-1,self.E, self.E)                # (B, E*E) -> (B, E,E)
        b = self.fc_b(y)[:,None,:]                              # (B, E)   -> (B, 1,E)

        y = nn.functional.gelu(torch.bmm(x, w) + b)                      # (B,T,E) @ (B,E,E) + (B,1,E) = (B,T,E)
        y = y + x * self.w_1                                    # (B,T, E)
        #y = gelu(y)

        x = self.ln_2(y)                                        # (B,T, E)
        y = self.mlp_2(x)
        y = y  + x * self.w_2                                   # (B,T, E)
        #y = gelu(y)
        return y                                                # (B,T, E)

#===============================================================================
