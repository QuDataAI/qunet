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

#========================================================================================

def gelu(x):
    """
    Implementation of the GELU activation function currently in BERT and  GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5*x*(1.0 + torch.tanh(math.sqrt(2.0/math.pi)*(x+0.044715*torch.pow(x,3.0))))

#========================================================================================

class MLP(nn.Module):
    """
    Полносвязная сеть с одним скрытым слоем: (B,*, input) ->  (B,*, output)
    """
    def __init__(self, cfg) -> None:
        super().__init__()
        cfg = cfg or {
            'input'   : 3,       # число входов
            'hidden'  : None,    # число нейронов в скрытом слое
            'stretch' : 4,       # если есть, то hidden = int(stretch*input)
            'output'  : 1,       # число выходов
            'fun'     : 'gelu',  # активационная функция: gelu, relu, sigmoid, tanh
            'drop'    : 0,       # dropout на выходе
        }
        cfg   = copy.deepcopy(cfg)
        assert 'hidden' in cfg or 'stretch' in cfg, 'no hidden/stretch in MLP: {cfg}'
        assert cfg['input']>0  and cfg['output']>0, f"worng input/output in MLP: {cfg}"

        if 'stretch' in cfg and cfg['stretch'] > 0:
            cfg['hidden'] = int(cfg['stretch'] * cfg['input'])
        assert cfg['hidden'] > 0, f"worng hidden in MLP cfg: {cfg}: {type(cfg['hidden'])}"

        cfg['fun']  = cfg.get('fun',  'gelu')      #  значения по умолчанию
        cfg['drop'] = cfg.get('drop', 0)
        self.cfg    = cfg

        self.create()

    def create(self):
        cfg=self.cfg

        if   cfg['fun'] == 'gelu':    self.fun = gelu
        elif cfg['fun'] == 'relu':    self.fun = nn.functional.relu
        elif cfg['fun'] == 'sigmoid': self.fun = torch.sigmoid
        elif cfg['fun'] == 'tanh':    self.fun = torch.tanh

        self.fc1  = nn.Linear(cfg['input'],  cfg['hidden'])
        self.fc2  = nn.Linear(cfg['hidden'], cfg['output'])
        self.drop = nn.Dropout(cfg['drop']) if cfg['drop'] > 0 else None

    def forward(self, x):
        x = self.fc1(x)
        x = self.fun(x)
        x = self.fc2(x)
        if self.drop is not None:
            x = self.drop(x)
        return x

#========================================================================================

class CNN(nn.Module):
    """
    Свёрточная сеть: (B, C, H, W) ->  (B, C', H', W')
    """
    def __init__(self, cfg) -> None:
        super().__init__()
        cfg = cfg or {
            'input'    : (1, 64, 64),   # входной тензор: (channels, height, width)
            'channel'  : [16,32,64],    # число каналов в каждом слое
            'kernel'   : [3,3,3],       # int or list: размеры конволюционного ядра
            'stride'   : 1,             # int or list: шаг конволюционного ядра
            'padding'  : 1,             # int or list: забивка вокруг картинки
            'pool_ker' : 2,             # int or list: ядро max-пулинга
            'pool_str' : 2,             # int or list: шаг max-пулинга
            'drop'     : 0,             # int or list: dropout после каждого слоя
            'output'   : None           # выходной тензор устанавливает create()
        }
        cfg   = copy.deepcopy(cfg)
        n = len(cfg['channel'])
        if type(cfg['drop'])     == int: cfg['drop']     = [cfg['drop']]    * n
        if type(cfg['kernel'])   == int: cfg['kernel']   = [cfg['kernel']]  * n
        if type(cfg['stride'])   == int: cfg['stride']   = [cfg['stride']]  * n
        if type(cfg['padding'])  == int: cfg['padding']  = [cfg['padding']] * n
        if type(cfg['pool_ker']) == int: cfg['pool_ker'] = [cfg['pool_ker']]* n
        if type(cfg['pool_str']) == int: cfg['pool_str'] = [cfg['pool_str']]* n
        self.cfg   = cfg
        self.create()

    def forward(self, x):
        x = self.model(x)
        return x

    def create(self):
        c, w, h  =  self.cfg['input']
        channels = [ c ] + self.cfg['channel']
        layers = []
        for i in range(len(channels)-1):
            kernel, stride     = self.cfg['kernel']  [i], self.cfg['stride'][i]
            padding            = self.cfg['padding'] [i]
            pool_ker, pool_str = self.cfg['pool_ker'][i], self.cfg['pool_str'][i]

            layers +=  [
                nn.Conv2d(channels[i],channels[i+1], kernel_size=kernel, stride=stride, padding=padding),
                nn.ReLU()]
            h = int( (h + 2*padding - kernel) / stride + 1)
            w = int( (w + 2*padding - kernel) / stride + 1)

            if pool_ker > 1:
                layers += [ nn.MaxPool2d(kernel_size=pool_ker, stride=pool_str) ]
                h = int( (h - pool_ker) / pool_str + 1)
                w = int( (w - pool_ker) / pool_str + 1)

            if self.cfg['drop'][i] > 0:
                layers += [ nn.Dropout(p=self.cfg['drop'][i]) ]

        self.cfg['output'] =  (channels[-1], w, h)
        self.model =  nn.Sequential(*layers)

#===============================================================================

class SelfAttention(nn.Module):
    """
    Основан на: https://github.com/karpathy/nanoGPT/blob/master/model.py
    """
    def __init__(self, cfg):
        # cfg: emb, heads, tokens, causal, drop_attn
        super().__init__()
        cfg = cfg or {
            'emb'    : 64,      # размерность эмбедига
            'heads'  : 8,       # число голов (emb % heads == 0 !)
            'causal' : False,   # причинное внимание (как в GPT) иначе как в BERT
            'tokens' : 2048,    # максимальное число тоенов (нужно для causal==True)
            'drop'   : 0,       # dropout на выходе внимания
        }
        cfg = copy.deepcopy(cfg)
        assert 'emb' in cfg and 'heads' in cfg, f"must be emb and heads in cfg:{cfg}"
        assert cfg['emb'] % cfg['heads'] == 0,  "emb must be div by heads!"
        cfg['causal'] = cfg.get('causal', False)
        cfg['tokens'] = cfg.get('tokens', 2048)
        cfg['drop']   = cfg.get('drop',   0)
        self.cfg   = cfg
        self.create()

    def create(self):
        cfg = self.cfg
        T,E = cfg['tokens'], cfg['emb']

        self.c_attn = nn.Linear(E,3*E)  # key, query, value projections for all heads
        self.c_proj = nn.Linear(E,  E)  # output projection

        self.attn_dropout  = nn.Dropout(cfg['drop'])      # regularization
        self.resid_dropout = nn.Dropout(cfg['drop'])

        if cfg['causal']:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(T, T)).view(1,1,T,T))
        self.heads  = cfg['heads']
        self.emb    = cfg['emb']
        self.causal = cfg['causal']

    def forward(self, x):  # (B,T,E)
        B,T,E = x.size() # batch size, sequence length, embedding dimensionality (emb)
        assert E == self.emb, "wrong input"

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q,k,v  = self.c_attn(x).split(self.emb, dim=2)
        k = k.view(B,T, self.heads, E // self.heads).transpose(1,2) # (B, nh, T, hs) hs = E/nh
        q = q.view(B,T, self.heads, E // self.heads).transpose(1,2) # (B, nh, T, hs)
        v = v.view(B,T, self.heads, E // self.heads).transpose(1,2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs)x(B, nh, hs, T) -> (B, nh, T,T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.causal:  # заполняем верхний угол треугольной матрицы -inf
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v                          # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B,T,E) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y            # (B,T,E)

#===============================================================================

class  TransformerBlock(nn.Module):
    def __init__(self, cfg, created=True):
        super().__init__()
        cfg = cfg or {
            'emb'             : 64,      # размерность эмбедига
            'res'             : 1,       # skip-connections (0-нет, 1-обычные, 2-тренеруемый, 3-покомпонентно)
            'att': {
                'heads'       : 8,       # число голов (emb % heads == 0 !)
                'causal'      : False,   # причинное внимание (как в GPT) иначе как в BERT
                'tokens'      : 1024,    # максимальное число тоенов (нужно для causal==True)
                'drop'        : 0,       # dropout на выходе внимания
            },
            'mlp': {
                'stretch'     : 4,       # увеличение от эбединга скрытого слоя (2,4,..): hidden=emb*scale
                'drop'        : 0,       # dropout на выходе mlp
                'fun'         : 'gelu',  # активационная MLP-функция: 'gelu','relu','sigmoid','tanh'
            }
        }
        cfg   = copy.deepcopy(cfg)
        assert 'emb' in cfg,  "Must be emb in TransformerBlock cfg"
        cfg['res']            = cfg.get('res', 2)
        cfg['att']            = cfg.get('att', {})
        cfg['att']['emb']     = cfg['emb']
        cfg['att']['drop']    = cfg['att'].get('drop',  0)

        cfg['mlp']            = cfg.get('mlp', {})
        cfg['mlp']['input']   = cfg['emb']
        cfg['mlp']['output']  = cfg['emb']
        cfg['mlp']['stretch'] = cfg['mlp'].get('stretch', 4)
        cfg['mlp']['fun']     = cfg['mlp'].get('fun',  'gelu')
        cfg['mlp']['drop']    = cfg['mlp'].get('drop',  0)

        self.cfg = cfg
        if created:
            self.create()

    def create(self):
        cfg = self.cfg
        self.ln_1 = nn.LayerNorm (cfg['emb'])
        self.attn = SelfAttention(cfg['att'])

        self.mlp = None
        if cfg['mlp']['stretch'] > 0:
            self.ln_2 = nn.LayerNorm(cfg['emb'])
            self.mlp  = MLP(cfg['mlp'])

        if   cfg['res'] == 3:     # train multiplier for each components
            self.w_att = nn.Parameter( torch.ones( cfg['emb'] ) )
            self.w_mlp = nn.Parameter( torch.ones( cfg['emb'] ) )
        elif cfg['res'] == 2:     # train common multiplier
            self.w_att   = nn.Parameter( torch.ones(1) )
            self.w_mlp   = nn.Parameter( torch.ones(1) )
        else:                     # constant multiplayer
            self.w_att   = self.register_buffer("w_att", torch.Tensor(cfg['res']))
            self.w_mlp   = self.register_buffer("w_mlp", torch.Tensor(cfg['res']))

    def forward(self, x):                          # (B,T,E)
        """
        Классический highway: w=sigmoid(linear(x)); x*w + (1-w)*mlp, ...
        не работал для нейтрино. Возможно это негативный эффект постоянного
        умножения x (и потом градиента) на число меньшее 1:  x*0.5*0.5*...
        Не работал также linear(x) + mlp, с наяальной единичной матрицей :(
        """
        x = x * self.w_att + self.attn(self.ln_1(x))
        x = x * self.w_mlp + self.mlp (self.ln_2(x))
        return x                                  # (B,T,E)

#===============================================================================

class  PointsBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg = cfg or {
            'emb'             : 64,      # размерность эмбедига
            'max'             : False,
            'mean'            : True,
            'res'             : 1,       # residual петли (0-нет, 1-обычные, 2-тренеруемые)
            'mlp1': {
                'stretch'     : 4,       # увеличение от эбединга скрытого слоя (2,4,..): hidden=emb*scale
                'drop'        : 0,       # dropout на выходе mlp
            },
            'mlp2': {
                'stretch'     : 4,       # увеличение от эбединга скрытого слоя (2,4,..): hidden=emb*scale
                'drop'        : 0,       # dropout на выходе mlp
            }
        }
        cfg   = copy.deepcopy(cfg)

        cfg['mean'] = cfg.get('mean', True)
        cfg['max']  = cfg.get('max', False)
        cfg['res']  = cfg.get('res', 2)
        assert cfg['mean'] or cfg['max'], f"PointsBlock need mean or/and max, cfg={cfg}"

        cfg['mlp1'] = cfg.get('mlp1', {})
        cfg['mlp2'] = cfg.get('mlp2', {})
        cfg['mlp1']['stretch'] = cfg['mlp1'].get('stretch', 4)
        cfg['mlp2']['stretch'] = cfg['mlp2'].get('stretch', 4)
        self.cfg = cfg
        self.create()

    def create(self):
        cfg = self.cfg
        E, E2 = cfg['emb'], cfg['emb']
        self.ln_1 = nn.LayerNorm(E)
        self.ln_2 = nn.LayerNorm(E)

        n_cat = 2 if cfg['max'] and cfg['mean'] else 1
        self.mlp_1 = MLP(dict(input=E, stretch=cfg['mlp1']['stretch'], output=E2))
        self.fc_w  = nn.Linear(n_cat*E2, E*E)
        self.fc_b  = nn.Linear(n_cat*E2, E)
        self.mlp_2 = MLP(dict(input=E, stretch=cfg['mlp1']['stretch'], output=E))

        if cfg['res'] == 2:
            self.w_1   = nn.Parameter( torch.ones(1) )
            self.w_2   = nn.Parameter( torch.ones(1) )
        else:
            self.w_1   = cfg['res']
            self.w_2   = cfg['res']

        self.E       = E
        self.is_max  = cfg['max']
        self.is_mean = cfg['mean']


    def forward(self, x):                                       # (B,T, E)
        x = self.ln_1(x)                                        # (B,T, E)
        y = self.mlp_1(x)                                       # (B,T, E')
        agg = []
        if self.is_mean:
            agg.append( y.mean(dim=1) )
        if self.is_max:
            agg.append( y.max(dim=1)[0] )
        y = torch.cat(agg, dim=1)                               # (B, n_cat*E')
        w = self.fc_w(y).view(-1,self.E, self.E)                # (B, E*E) -> (B, E,E)
        b = self.fc_b(y)[:,None,:]                              # (B, E)   -> (B, 1,E)

        y = gelu(torch.bmm(x, w) + b)                           # (B,T,E) @ (B,E,E) + (B,1,E) = (B,T,E)
        y = y + x * self.w_1                                    # (B,T, E)
        #y = gelu(y)

        x = self.ln_2(y)                                        # (B,T, E)
        y = self.mlp_2(x)
        y = y  + x * self.w_2                                   # (B,T, E)
        #y = gelu(y)
        return y                                                # (B,T, E)

