import math, copy,  re
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn

from ..config           import Config
from .total             import Create
from .mlp               import MLP
from .residual          import Residual
from .transformer_plot  import plot_transformer_blocks
from ..modelstate       import ModelState

    
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
            drop (int=0):
                is or not dropout after softmax
            res (int=1):
                kind of skip-connections (for TransformerBlock): (1) x+f(x), (2)  x+w*f(x) - w training one for all E; (3) w training for each E
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
            drop   = 0,        # is or not dropout after softmax
        ))

    #---------------------------------------------------------------------------

    def create(self):
        cfg = self.cfg
        assert cfg.E is not None and cfg.E > 0 and cfg.H > 0, f"must be E and H in cfg:{cfg.get_str()}"
        assert cfg.E % cfg.H == 0,  "E must be div by H: {cfg.E} % {cfg.H} != 0"

        T,E = cfg.T_max, cfg.E

        self.c_attn = nn.Linear(E,3*E)  # key, query, value projections for all heads
        self.c_proj = nn.Linear(E,  E)  # output projection
        
        self.drop  = nn.Dropout(0.0) if cfg.drop else nn.Identity()

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
#                              Transformer
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
            H (int = 1):
                number of heads E % H == 0 !
            causal (bool=False):
                kind of causal attention mask (True: GPT, False: Bert)
            T_max  (int=2048):
                maximum number of tokens (needed for causal==True)
            mult (int=1):
                multiplier type scale: (0) - none: x+f(x), (1)-trainable scalar, (2)-trainable vector: x+s*f(x)
            scale (float = 1.):
                initial value of the learning residual multiplier
        Example:
        ------------
        ```
        m = Transformer(E=64, H=8, blocks="3(att mlp) 2(fft mlp) 5(mlp)")

        x = torch.rand(1,10,64)  # (B,T,E) = (batch, token, embedding)
        y = m(x)                 # (B,T,E) -> (B,T,E)
        ```
        """
        super().__init__()
        self.cfg = Transformer.default()
        cfg = self.cfg.set(*args, **kvargs)        
        self.create(cfg)

    #---------------------------------------------------------------------------

    @staticmethod
    def default():
        return copy.deepcopy(Config(
            E         = None,  # tokens embedding dim    
            blocks    = "",    # string with tokens
            H         = 1,          #  number of heads E % H == 0 !
            causal    = False,      # kind of causal attention mask (True: GPT, False: Bert)
            T_max     = 2048,       # maximum number of tokens (needed for causal==True)
            mult      = 0,          # multiplier type scale: (0) - none: x+f(x), (1)-trainable scalar, (2)-trainable vector: x+s*f(x)
            scale     = 1.,         # initial value of the learning residual multiplier
            
            drop_inside = 0,        # тип Dropout внутри Attention и MLP
            drop_after  = 0,        # тип Dropout после блоков ('r')

            shift_inside  = 0,      # добавлять ShiftFeatires между Conv2d в блоках ('r')
            shift_after   = 0,      # добавлять ShiftFeatires после блоков ('r')

            norm          = 2,      # тип нормировки для токенов 'n'
            norm_before   = 0,      # тип нормировки перед блоком block(x)
            norm_inside   = 2,      # тип нормировки внутри блока block(x) между Conv2d
            norm_block    = 2,      # тип нормировки внутри блока block(x) после последнего Conv2d
            norm_align    = 2,      # тип нормировки внутри блока align(x) после Conv2d
            norm_after    = 0,      # добавлять нормировку после блоков ('r')

            fun           = 'gelu', # активационная функция для токена `f`
            fun_inside    = 'gelu', # активационная функция для блоков ('r')
            fun_after     = "",     # добавлять активационную функцию после блоков ('r')

            dim           = 1,
        ))

    #---------------------------------------------------------------------------

    def create(self, cfg):
        assert cfg.E is not None and type(cfg.E)==int, f"Wrong E={cfg.input}, should be int (embedding dim)"     
        self.__num_blocks = 1
        tokens = cfg.blocks.replace('(', ' ( ').replace(')', ' ) ')
        tokens = cfg.blocks.replace('[', ' [ ').replace(']', ' ] ')
        tokens = tokens.split(' ')
        tokens = [token for token in tokens if token]
        all_blocks = self.parse(tokens, cfg)
        self.blocks = nn.ModuleList(all_blocks)

    #---------------------------------------------------------------------------

    def parse(self, tokens, cfg):
        E = cfg.E
        all_blocks, i = [], 0
        while  i < len(tokens):
            token = tokens[i]

            if token == '(':
                i1 = i+1
                i = self.get_close_bracket(tokens, i+1,  '(', ')')
                blocks = []
                for _ in range(self.__num_blocks):
                    b, E = self.get_sequential(tokens, i1, i, E, cfg=cfg)
                    blocks += [b]
                self.__num_blocks = 1
                all_blocks +=  blocks 
                i += 1
                continue

            if token == '[':
                i1 = i+1
                i = self.get_close_bracket(tokens, i+1,  '[', ']')
                blocks = []
                for _ in range(self.__num_blocks):
                    blks, E = self.get_sequential(tokens, i1, i, E, cfg=cfg)
                    blks = [ b for b in blks ]
                    blocks += blks
                self.__num_blocks = 1
                all_blocks +=  blocks 
                i += 1
                continue


            blocks, E = self.get_blocks(token=token, E=E, cfg=cfg)            
            if blocks  and type(blocks[0]) != nn.Identity:
                all_blocks += blocks

            i += 1
        return all_blocks
    
    #---------------------------------------------------------------------------

    def get_close_bracket(self, tokens, i, bra = '(', ket = ')'):
        open = 1
        while i < len(tokens):
            if   tokens[i] == bra:  open += 1
            elif tokens[i] == ket:  open -= 1
            if open == 0:
                return i
            i += 1
        assert tokens, f"No close bracket in {tokens}"

    #---------------------------------------------------------------------------

    def get_sequential(self, tokens, i1, i2, E, cfg):
        layers, i = [], i1
        while i < i2:
            token = tokens[i]
            if len(token):
                if token == '(':
                    i1 = i+1
                    i  = self.get_close_bracket(tokens, i+1,  '(', ')')
                    blocks, E = self.get_sequential(tokens, i1, i, E, cfg)
                else:
                    blocks, E = self.get_blocks(token=token, E=E, cfg=cfg)
                    if blocks is None: 
                        continue
                layers += blocks
            i += 1

        return nn.Sequential(*layers), E
    #---------------------------------------------------------------------------

    def get_blocks(self, token, E, cfg):
        if token[0].isdigit():
            num = re.search(r'(\d+)', token).group() # number of repetitions of the token            
            token = token[len(num):]
            num = int(num)            
            if len(token) == 0:
                self.__num_blocks = num
                return None, E
            else:
                blocks = []
                for _ in range(num):
                    b, E = self.get_blocks(token, E, cfg)
                    if b is None: 
                        continue
                    blocks += b
                return blocks, E

        if  token[:3] == 'mlp' and ( len(token) == 3 or token[3].isdigit() ):
            parts    = token[3:].split('_')  if len(token[3:]) > 1 else []
            E2       = int(parts[0]) if len(parts) > 0 else E
            stretch  = int(parts[1]) if len(parts) > 1 else 4
            block    = Residual(
                MLP(input=E, output=E2, stretch=stretch, drop = cfg.drop_inside, norm=cfg.norm_inside, fun=cfg.fun_inside),
                E=E, Eout=E2, mult = cfg.mult, scale=cfg.scale,
                norm_before= cfg.norm_before, norm_align = cfg.norm_align,  norm_block = cfg.norm_block,
                drop_after = cfg.drop_after, dim=1, shift_after=cfg.shift_after,
                norm_after = cfg.norm_after, fun_after = cfg.fun_after, name=token)
            block.name = token
            E = E2
            return [block], E

        if  token == 'att' and ( len(token) == 3 or token[3].isdigit() ):
            block = Residual(
                Attention(E=E, H=cfg.H, causal=cfg.causal, T_max=cfg.T_max, drop = cfg.drop_inside),
                E=E, Eout=E, mult = cfg.mult, scale=cfg.scale,
                norm_before= cfg.norm_before, norm_align = cfg.norm_align,  norm_block = cfg.norm_block,
                drop_after = cfg.drop_after, dim=1, shift_after=cfg.shift_after,
                norm_after = cfg.norm_after, fun_after = cfg.fun_after, name=token )
            block.name = token
            return [block], E

        if  token == 'fft' and ( len(token) == 3 or token[3].isdigit() ):
            block = Residual(
                FFT(),
                E=E, Eout=E, mult = cfg.mult, scale=cfg.scale,
                norm_before= cfg.norm_before, norm_align = cfg.norm_align, norm_block = cfg.norm_block,
                drop_after = cfg.drop_after, dim=1, shift_after=cfg.shift_after,
                norm_after = cfg.norm_after, fun_after = cfg.fun_after, name=token )
            block.name = token
            return [block], E
        
        if token[0] == 'n' and ( len(token) == 1 or token[1].isdigit() ):
            parts  = token[1:].split('_') if len(token) > 1 else []
            kind   = int(parts[0]) if len(parts) > 0 else cfg.norm            
            blocks = [ Create.norm(E, kind, 1) ] if kind else None
            return blocks, E

        if token == "f":
            if cfg.fun:
                return [Create.activation(cfg.fun)], E
            else:
                return None, E

        assert token in ['relu', 'gelu', 'relu6', 'sigmoid', 'tanh', 'swish', 'hswish', 'hsigmoid'], f"Unknown activation function '{token}'"
        return [Create.activation(token)], E

    #---------------------------------------------------------------------------

    def forward(self, x):
        """  (B,T,E) -> (B,T,E) """
        for block in self.blocks:
            x = block(x)                           # (B,T,E)
        return x

    #---------------------------------------------------------------------------

    def update(self):
        for block in self.blocks:
            if hasattr(block, "update"):            
                block.update()

    #---------------------------------------------------------------------------

    def decay(self):
        res = set()
        for block in self.blocks:
            if hasattr(block, "decay"):
                res.update(block.decay() )    
        return res        

    #---------------------------------------------------------------------------

    def debug(self, value=True, beta=None):
        for block in self.blocks:
            if hasattr(block, "debug"):
                block.debug(value, beta)

    #---------------------------------------------------------------------------

    def plot(self, w=12, h=3, eps=1e-8, bar_width = 0.25, info=False):
        plot_transformer_blocks(self.blocks, w=w, h=h, eps=eps, bar_width = bar_width, info=info)
        
    #---------------------------------------------------------------------------

    @staticmethod
    def unit_test():
        dim = 2
        if dim==3:
            B, T, E, H = 1, 10, 128, 16
            #cfg = Config(E=E, H=H, blocks="2[att mlp] fft mlp")
            cfg = Config(E=E, H=H, blocks="mlp")
            m = Transformer(cfg)
            x = torch.rand(B,T,E)  # (B,T,E) = (batch, token, embedding)
            y = m(x)                                # (B,T,E) -> (B,T,E)
            state = ModelState(m)
            state.layers(2, input_size=(B,T,E))
        else:
            B, E = 10, 128            
            cfg = Config(E=E,  blocks="mlp256 mlp")
            m = Transformer(cfg)            
            x = torch.rand(B,E)
            y = m(x)                                # (B,E) -> (B,E)
            state = ModelState(m)
            state.layers(2, input_size=(B,E))

        """
        m.debug(True)        
        y = m(x)               # (B,T,E) -> (B,T,E)
        y.mean().backward()
        m.update()
        #print( m.plot(info=True) )
        #for block in m.blocks: print(block.cfg)
        """

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
