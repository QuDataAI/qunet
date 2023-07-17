import copy, re
import numpy as np, matplotlib.pyplot as plt, matplotlib.ticker as ticker
import torch, torch.nn as nn

from ..config      import Config
from ..modelstate  import ModelState
from .total        import Create,  ShiftFeatures
from .residual     import Residual

#===============================================================================

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=False, norm=1, fun="relu"):
        """
        Simple convolution block
        """
        padding = (kernel_size - 1) // 2
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)]

        norm = Create.norm(out_channels, norm, 2)
        if type(norm) != nn.Identity:
            layers.append(  norm )

        if fun:
            layers.append( Create.activation(fun) )

        super().__init__(*layers)

#===============================================================================

class Conv2dBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3,  bias=False, stride=1, stride_first=True,
                drop=0, shift=0, norm=1, norm_block=1, num=2,  fun="relu"):
        """
        shape and channels will not change
        """
        padding = (kernel_size - 1) // 2
        layers  = []
        for i in range(num):
            if (stride_first and i==0) or (not stride_first and i+1==num):
                s = stride            
            else:
                s = 1
            if i > 0:
                stride = 1
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=s, padding=padding, bias=bias) )
            in_channels = out_channels

            if i+1 < num:
                if norm:             
                    layers.append( Create.norm(out_channels, norm, 2) )
                if fun:
                    layers.append( Create.activation(fun) )
                if drop:
                    layers.append( Create.dropout(drop) )
                if shift:
                    layers.append( ShiftFeatures() )

        if norm_block:
            layers.append( Create.norm(out_channels, norm_block, 2) )

        super().__init__(*layers)

#===============================================================================

class CNN(nn.Module):
    def __init__(self,  *args, **kvargs) -> None:
        """
        Conv2d: c[channels]_[kernel]_[stride]  padding = (kern - 1) // 2
            c64      - Conv2d(..., 64, kernel_size=3, padding=1)
            c64_5    - Conv2d(..., 64, kernel_size=5, padding=2)
            c64_7_2  - Conv2d(..., 64, kernel_size=7, padding=3, stride=2)

        MaxPool2d: m[kernel]_[stride]
            m        - nn.MaxPool2d(kernel=2, stride=2)
            m3_2     - nn.MaxPool2d(kernel=3, stride=2, padding=1)

        AverPool2d: a[kernel]_[stride]
            a        - nn.AvgPool2d(kernel=2, stride=2)
            a3_2     - nn.AvgPool2d(kernel=3, stride=2, padding=1)

        Norm2d: n[kind]
            n        - BatchNorm2d
            n1       - BatchNorm2d
            n2       - InstanceNorm2d

        fun activation:
            f        - current activation from cfg.fun
            relu ... - see Create.activation in total.py

        Residual:  r[channels]_[kernel]_[stride]
            r        - Residual (don't change size of immage and channels)
            r128     - Residual with change number of channels
            r128_3_2 - kernel 3 and stride = 2 in first Conv2s layer (like resnetXX)

        Block:  rNo resudial [channels]_[kernel]_[stride]
            b        - don't change size of immage and channels
            b128     - change number of channels
            b128_3_2 - kernel 3 and stride = 2 in first Conv2s layer

        Dropout            
            d1   -  Dropout1d
            d2   -  Dropout2d
            d    -  DropoutXd, X = cfg.drop

        """
        super().__init__()
        self.cfg = CNN.default()
        cfg = self.cfg.set(*args, **kvargs)
        self.create(cfg)

    #---------------------------------------------------------------------------

    @staticmethod
    def default():
        return copy.deepcopy(Config(
            input  = 3,             # число каналов на входе
            blocks = "",            # строка со списком токенов, задающих последовательность блоков
            bias   = False,         # слои Conv2d создавать со смещением
            
            stride_first  = True,   # делать stride>1 на первом Conv2d, иначе на последнем

            drop = 2,             # тип Dropout для всех токенов `d` строки blocks
            drop_inside = 2,      # тип Dropout между Conv2d в блоках ('r', 'b')
            drop_after  = 2,      # тип Dropout после блоков ('r')

            shift_inside  = 0,      # добавлять ShiftFeatires между Conv2d в блоках ('r')
            shift_after   = 0,      # добавлять ShiftFeatires после блоков ('r')

            norm   = 1,             # тип нормировки для токенов 'n'

            norm_before   = 0,      # тип нормировки перед блоком block(x)
            norm_inside   = 1,      # тип нормировки внутри блока block(x) между Conv2d
            norm_block    = 1,      # тип нормировки внутри блока block(x) после последнего Conv2d
            norm_align    = 1,      # тип нормировки внутри блока align(x) после Conv2d
            norm_after    = 0,      # добавлять нормировку после блоков ('r')
            
            fun           = 'relu', # активационная функция для токена `f`
            fun_inside    = 'relu', # активационная функция для блоков ('r')
            fun_after     = "",     # добавлять активационную функцию после блоков ('r')
            
            mult          = 0,      # тип множителя scale (0-нет, 1-обучаемый скаляр, 2-обучаемый вектор)
            scale         = 1.,     # начальное значение scale (для mult > 0)

            pool          = 1,      # добавить после blocks слой nn.AdaptiveAvgPool2d(1) или nn.AdaptiveMaxPool2d(1) если 2
            flat          = True,   # добавить в конце nn.Flatten() 
        ))

    #---------------------------------------------------------------------------

    def create(self, cfg):
        assert cfg.input is not None and type(cfg.input)==int, f"Wrong cfg.input={cfg.input}, should be int (number of input channels)"
        channels = cfg.input

        tokens = cfg.blocks.replace('(', ' ( ').replace(')', ' ) ').split(' ')
        tokens = [token for token in tokens if token]
        all_blocks, i = [], 0
        while  i < len(tokens):
            token = tokens[i]

            if token == '(':
                i1 = i+1
                i = self.get_close_bracket(tokens, i+1,  '(', ')')
                blocks, channels = self.get_sequential(tokens, i1, i, channels, cfg=cfg)
                all_blocks += [ blocks ]
                i += 1
                continue

            blocks, channels = self.get_blocks(token=token, channels=channels, cfg=cfg)
            if blocks  and type(blocks[0]) != nn.Identity:
                all_blocks += blocks

            i += 1

        if cfg.pool==1:
            block = nn.AdaptiveAvgPool2d(1)
            block.name = 'avg'
            all_blocks.append( block )
        elif cfg.pool==2:
            block = nn.AdaptiveMaxPool2d(1)
            block.name = 'max'
            all_blocks.append( block )
            
        if cfg.flat:
            block = nn.Flatten()
            block.name = 'flat'
            all_blocks.append( block )

        self.blocks = nn.ModuleList(all_blocks)

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

    def get_sequential(self, tokens, i1, i2, channels, cfg):
        layers, i = [], i1
        while i < i2:
            token = tokens[i]
            if len(token):
                if token == '(':
                    i1 = i+1
                    i  = self.get_close_bracket(tokens, i+1,  '(', ')')
                    blocks, channels = self.get_sequential(tokens, i1, i, channels, cfg)
                else:
                    blocks, channels = self.get_blocks(token=token, channels=channels, cfg=cfg)
                layers += blocks
            i += 1

        return nn.Sequential(*layers), channels
    #---------------------------------------------------------------------------

    def get_blocks(self, token, channels, cfg):
        if token[0].isdigit():
            num = re.search(r'(\d+)', token).group() # number of repetitions of the token
            token = token[len(num):]
            num = int(num)            
            blocks = []
            for _ in range(num):
                b, channels = self.get_blocks(token, channels, cfg)
                blocks += b
            return blocks, channels

        if  token[:3] == 'cnf' and ( len(token) == 3 or token[3].isdigit() ):
            parts  = token[3:].split('_')  if len(token[3:]) > 0 else []
            chan   = int(parts[0]) if len(parts) > 0 else channels
            kern   = int(parts[1]) if len(parts) > 1 else 3
            stride = int(parts[2]) if len(parts) > 2 else 1
            pad    = int(parts[3]) if len(parts) > 3 else (kern - 1) // 2
            layers = [nn.Conv2d(channels, chan, kern, stride=stride,  padding=pad, bias=cfg.bias)]
            if cfg.norm:
                layers.append(Create.norm(chan, cfg.norm, 2))
            if cfg.fun:
                layers.append(Create.activation(cfg.fun) )
            block = nn.Sequential(*layers)
            block.name = token
            channels = chan
            return [block], channels

        if  token[0] == 'c' and ( len(token) == 1 or token[1].isdigit() ):
            parts  = token[1:].split('_')  if len(token) > 1 else []
            chan   = int(parts[0]) if len(parts) > 0 else channels
            kern   = int(parts[1]) if len(parts) > 1 else 3
            stride = int(parts[2]) if len(parts) > 2 else 1
            pad    = int(parts[3]) if len(parts) > 3 else (kern - 1) // 2
            block = nn.Conv2d(channels, chan, kern, stride=stride, padding=pad,  bias=cfg.bias)
            block.name = token
            channels = chan
            return [block], channels

        if token[0] == 'm' and ( len(token) == 1 or token[1].isdigit() ):
            parts  = token[1:].split('_')  if len(token) > 1 else []
            kern   = int(parts[0]) if len(parts) > 0 else 2
            stride = int(parts[1]) if len(parts) > 1 else 2
            pad    = int(parts[2]) if len(parts) > 2 else 0            
            block = nn.MaxPool2d(kernel_size=kern,stride=stride, padding=pad)
            block.name = token
            return [block], channels

        if token[0] == 'a' and ( len(token) == 1 or token[1].isdigit() ):
            parts  = token[1:].split('_')  if len(token) > 1 else []
            kern   = int(parts[0]) if len(parts) > 0 else 2
            stride = int(parts[1]) if len(parts) > 1 else 2
            pad    = int(parts[2]) if len(parts) > 2 else 0
            block = nn.AvgPool2d(kernel_size=kern,stride=stride, padding=pad)
            block.name = token
            return [block], channels

        if token[0] == 'n' and ( len(token) == 1 or token[1].isdigit() ):
            parts  = token[1:].split('_') if len(token) > 1 else []
            kind   = int(parts[0]) if len(parts) > 0 else cfg.norm            
            blocks = [ Create.norm(channels, kind, 2) ] if kind else None
            return blocks, channels

        if token[0] == 'r' and ( len(token) == 1 or token[1].isdigit() ):
            parts  = token[1:].split('_')  if len(token) > 1 else []
            Eout   = int(parts[0]) if len(parts) > 0 else channels
            kern   = int(parts[1]) if len(parts) > 1 else 3
            stride = int(parts[2]) if len(parts) > 2 else 1

            block= Residual(
                    Conv2dBlock(channels, Eout, kernel_size=kern, stride=stride, stride_first=cfg.stride_first, bias=cfg.bias, norm=cfg.norm_inside, norm_block=cfg.norm_block, fun=cfg.fun_inside, drop=cfg.drop_inside, shift=cfg.shift_inside),
                    E=channels, Eout=Eout, mult = cfg.mult, scale=cfg.scale, stride=stride,
                    norm_before= cfg.norm_before,    # для resnetXX 0
                    norm_align = 0 if (channels==Eout and stride==1) else cfg.norm_align,
                    drop_after = cfg.drop_after, dim=2, shift_after=cfg.shift_after,
                    norm_after = cfg.norm_after, fun_after = cfg.fun_after,
                    name=token )
            channels = Eout
            return [block], channels

        if token[0] == 'b' and ( len(token) == 1 or token[1].isdigit() ):
            parts  = token[1:].split('_')  if len(token) > 1 else []
            Eout   = int(parts[0]) if len(parts) > 0 else channels
            kern   = int(parts[1]) if len(parts) > 1 else 3
            stride = int(parts[2]) if len(parts) > 2 else 1

            block = Conv2dBlock(channels, Eout, kernel_size=kern, stride=stride, stride_first=cfg.stride_first, bias=cfg.bias, norm=cfg.norm_inside, norm_block=cfg.norm_block, fun=cfg.fun, drop=cfg.drop_inside, shift=cfg.shift_inside)
            block.name = token
            channels = Eout
            return [block], channels

        if  token[0] == 'd' and ( len(token) == 1 or token[1].isdigit() ):
            parts = token[1:].split('_') if len(token) > 1 else []
            dim  = int(parts[0]) if len(parts) > 0 else cfg.drop
            assert dim in [1,2], f"Dropout layer dimension can be 1 or 2, but got {dim}"
            block = Create.dropout(dim) 
            block.name = token
            return [block], channels

        if  token[0] == 's' and ( len(token) == 1 or token[1].isdigit() ):
            block = ShiftFeatures()
            block.name = token
            return [block], channels

        if token == "f":
            if cfg.fun:
                return [Create.activation(cfg.fun)], channels
            else:
                return None, channels

        assert token in ['relu', 'gelu', 'relu6', 'sigmoid', 'tanh', 'swish', 'hswish', 'hsigmoid'], f"Unknown activation function {token}"
        return [Create.activation(token)], channels

    #---------------------------------------------------------------------------

    def forward(self, x):
        """  (B,C,H,W) -> (B,C',H',W') """
        for block in self.blocks:
            x = block(x)
        return x

    #---------------------------------------------------------------------------

    def update(self):
        for block in self.blocks:
            if  hasattr(block, "update"):
                block.update()

    #---------------------------------------------------------------------------

    def debug(self, value=True, beta=None):
        for block in self.blocks:
            if  hasattr(block, "debug"):
                block.debug(value, beta)

    #---------------------------------------------------------------------------

    def set_block_drops(self, drop=None, drop_b=None, std=None, p=None):
        num_res = sum([1 for block in self.blocks if type(block) == Residual])

        drop   = [drop]  *num_res if drop   is None or type(drop) in [int, float] else drop
        drop_b = [drop_b]*num_res if drop_b is None or type(drop_b) in [int, float] else drop_b
        std    = [std]   *num_res if std is None or type(std) in [int, float] else std
        p      = [  p]   *num_res if   p is None or type(p)   in [int, float] else p

        i=0
        for block in self.blocks:
            if type(block) == Residual:
                block.set_drop(drop[i], drop_b[i], std[i], p[i])
                i += 1


    #---------------------------------------------------------------------------
    @staticmethod
    def resnet18():
        """resnet18
        ```
        # Equal:
        from torchvision.models import resnet18
        model = resnet18()
        ```
        """
        cfg = CNN.default()
        cfg(
            input       = 3,
            blocks      = "(cnf64_7_2 m3_2_1) 2r r128_3_2 r r256_3_2 r r512_3_2 r",
            norm_before = 0,
            norm_inside = 1,            
            norm_block  = 1,
            norm_after  = 0,
            fun         = 'relu',
            mult        = 0,                        
        )
        return cfg

    #---------------------------------------------------------------------------

    @staticmethod
    def unit_test():
        B,C,H,W = 1, 1, 28,28
        cnn = CNN(input = C, blocks = "(c64_7_2 n f m3_2) r64 m r128 m")
        x = torch.rand(B,C,H,W)
        y = cnn(x)
        s = ModelState(cnn)
        s.layers(2, input_size=(B,C,H,W))
        #print(cnn)
        print(f"ok CNN, output = {tuple(y.shape)}")

        return True

    #---------------------------------------------------------------------------

    def plot(self, w=12, h=3, eps=1e-8, bar_width = 0.75, info=False):
        blocks = self.blocks        
        fig, ax = plt.subplots(1,1, figsize=(w, h))

        idx    = np.arange( len(blocks) )
        xticks = [ block.name if hasattr(block, 'name') else i  for i, block in enumerate(blocks) ]
        plt.xticks(idx, xticks)        
        ax.grid(ls=":")    
        ax.set_ylabel("dx/x");  # ax.set_xlabel("blocks");
        
        plt.text(0,0,f" std\n", ha='left', transform = ax.transAxes, fontsize=6, family="monospace")

        idx_res, res, idx_res0, res0 =  [], [], [], []
        for i,block in enumerate(blocks):
            if hasattr(block, 'sqr_dx') and block.sqr_dx is not None and block.sqr_x is not None:
                dx = (block.sqr_dx / (block.sqr_x+eps)).sqrt().cpu().item()
                idx_res.append( i )
                res.append( dx )
            else:
                idx_res0.append( i )
                res0.append( 0 )
        if res:
            ax.set_ylim(bottom=0, top=1.1*np.max(res))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.3f'))
            ax.bar(idx_res, res, width = bar_width, edgecolor ='grey', alpha=0.5)
        if res0:            
            ax.bar(idx_res0, res0, width = bar_width, edgecolor ='grey', alpha=0.5)

        ymin, ymax = ax.get_ylim()
        names = [''] * len(blocks)    
        for i,block in enumerate(blocks):
            if hasattr(block, 'name'):
                names[i] = block.name
                st = f"{block.std.cpu().item():.1f}" if hasattr(block, 'std') else ""
                plt.text(i, ymin, f"{st}\n", ha='center', fontsize=6, family="monospace")

        ax2 = ax.twinx()
        idx_scale_d, scale_d = [], []
        for i,block in enumerate(blocks):
            if hasattr(block, 'scale_d') and block.scale_d is not None:
                idx_scale_d.append(i)
                scale_d.append( block.scale_d.sqrt().cpu().item() )
        if scale_d:
            ax2.set_ylim(bottom=0, top=1.1*np.max(scale_d))
            ax2.plot(idx_scale_d, scale_d, marker=".")
        ax2.set_ylabel("scale")

        ax3 = ax.twinx()
        ax3.set_yscale('log')
        idx_scale_g, scale_g = [], []
        for i,block in enumerate(blocks):
            if hasattr(block, 'scale_g') and block.scale_g is not None:
                idx_scale_g.append(i)
                scale_g.append( block.scale_g.sqrt().cpu().item() )
        if scale_g:
            ax3.plot(idx_scale_g, scale_g, ":", marker=".")
        ax3.set_ylabel("scale grad")
        ax3.spines["right"].set_position(("outward", 50))
        
        plt.show()

        if info:
            return {'idx': idx, 'names': names, 'res': res, 'scale_d': scale_d, 'scale_g': scale_g}
