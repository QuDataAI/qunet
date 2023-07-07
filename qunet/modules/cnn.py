import copy, re
import numpy as np, matplotlib.pyplot as plt, matplotlib.ticker as ticker
import torch, torch.nn as nn

from ..config      import Config
from ..modelstate  import ModelState
from .total        import get_activation, get_norm, get_model_layers
from .residual     import Residual

#===============================================================================

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=False, norm=1, fun="relu"):
        """
        Simple convolution block
        """
        padding = (kernel_size - 1) // 2
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)]

        norm = get_norm(out_channels, norm, 2)
        if type(norm) != nn.Identity:
            layers.append(  norm )

        if fun:
            layers.append( get_activation(fun) )

        super().__init__(*layers)

#===============================================================================

class ResidualConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3,  bias=False, stride=1, norm=1, num = 2,  fun="relu"):
        """
        shape and channels will not change
        """
        padding = (kernel_size - 1) // 2
        layers  = []
        for i in range(num):
            if i > 0:
                stride = 1
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias) )
            in_channels = out_channels

            norm = get_norm(out_channels, norm, 2)
            if type(norm) != nn.Identity:
                layers.append(  norm )

            if i+1 < num and fun:
                layers.append( get_activation(fun) )

        super().__init__(*layers)

#===============================================================================

class DownPoolBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3,  bias=False,  norm=1, fun="relu", drop=0.):
        super().__init__(
            ConvBlock(in_channels, out_channels, kernel, bias=bias, norm=norm, fun=fun),
            nn.MaxPool2d(2),
            nn.Dropout2d(drop)
        )

#===============================================================================

class DownStrideBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=2, bias=False,  norm=1, fun="relu", drop=0.):
        super().__init__(
            ConvBlock(in_channels, out_channels, kernel=kernel, stride=2, bias=bias, norm=norm, fun=fun),
            nn.Dropout2d(drop)
        )

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
            relu ... - see get_activation in total.py

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
            d    -  DropoutXd, X = cfg.drop_d 

        """
        super().__init__()
        self.cfg = CNN.default()
        cfg = self.cfg.set(*args, **kvargs)

        assert cfg.input is not None and type(cfg.input)==int, f"Wrong cfg.input={cfg.input}, should be int (number of input channels)"
        channels = cfg.input

        tokens = cfg.blocks.replace('(', ' ( ').replace(')', ' ) ').split(' ')
        all_blocks, i = [], 0
        while  i < len(tokens):
            token = tokens[i]
            if len(token) == 0:
                i += 1
                continue

            if token == '(':
                i1 = i+1
                i = self.get_close_bracket(tokens, i+1,  '(', ')')
                blocks, channels = self.get_sequential(tokens, i1, i, channels, cfg=cfg)
                all_blocks +=  blocks
                i += 1
                continue

            blocks, channels = self.get_blocks(token=token, channels=channels, cfg=cfg)
            if blocks  and type(blocks[0]) != nn.Identity:
                all_blocks += blocks

            i += 1

        if cfg.avg:
            all_blocks.append( nn.AdaptiveAvgPool2d(1) )
        if cfg.flat:
            all_blocks.append( nn.Flatten() )

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
            print(num,  token)
            blocks = []
            for _ in range(num):
                b, channels = self.get_blocks(token, channels, cfg)
                blocks += b
            return blocks, channels

        if  token[:3] == 'cnf' and ( len(token) == 3 or token[3].isdigit() ):
            parts  = token[3:].split('_')  if len(token) > 1 else []
            chan   = int(parts[0]) if len(parts) > 0 else channels
            kern   = int(parts[1]) if len(parts) > 1 else 3
            stride = int(parts[2]) if len(parts) > 2 else 1
            pad    = int(parts[3]) if len(parts) > 3 else (kern - 1) // 2
            layers = [nn.Conv2d(channels, chan, kern, stride=stride,  padding=pad, bias=cfg.bias)]
            if cfg.norm:
                layers.append(get_norm(chan, cfg.norm, 2))
            if cfg.fun:
                layers.append(get_activation(cfg.fun) )
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
            blocks = [ get_norm(channels, kind, 2) ] if kind else None
            return blocks, channels

        if token[0] == 'r' and ( len(token) == 1 or token[1].isdigit() ):
            parts  = token[1:].split('_')  if len(token) > 1 else []
            Eout   = int(parts[0]) if len(parts) > 0 else channels
            kern   = int(parts[1]) if len(parts) > 1 else 3
            stride = int(parts[2]) if len(parts) > 2 else 1

            block= Residual(
                    ResidualConvBlock(channels, Eout, kernel_size=kern, stride=stride, bias=cfg.bias, norm=cfg.norm, fun=cfg.fun),
                    E=channels, Eout=Eout, res = cfg.res, gamma=cfg.gamma, stride=stride,
                    norm_before=0, norm_after = 0,   # подстраиваемся под resnetXX
                    norm_align = 0 if channels==Eout else cfg.norm,
                    dim=2, name=token )
            channels = Eout
            return [block], channels

        if token[0] == 'b' and ( len(token) == 1 or token[1].isdigit() ):
            parts  = token[1:].split('_')  if len(token) > 1 else []
            Eout   = int(parts[0]) if len(parts) > 0 else channels
            kern   = int(parts[1]) if len(parts) > 1 else 3
            stride = int(parts[2]) if len(parts) > 2 else 1

            block = ResidualConvBlock(channels, Eout, kernel_size=kern, stride=stride, bias=cfg.bias, norm=cfg.norm, fun=cfg.fun)
            block.name = token
            channels = Eout
            return [block], channels

        if  token[0] == 'd' and ( len(token) == 1 or token[1].isdigit() ):
            parts = token[1:].split('_') if len(token) > 1 else []
            dim  = int(parts[0]) if len(parts) > 0 else cfg.drop_d
            assert dim in [1,2], f"Dropout layer dimension can be 1 or 2, but got {dim}"
            block = nn.Dropout1d() if dim == 1  else  nn.Dropout2d()            
            block.name = token
            return [block], channels

        if token == "f":
            if cfg.fun:
                return [get_activation(cfg.fun)], channels
            else:
                return None, channels

        assert token in ['relu', 'gelu', 'relu6', 'sigmoid', 'tanh', 'swish', 'hswish', 'hsigmoid'], f"Unknown activation function {token}"
        return [get_activation(token)], channels

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

    def debug(self, value):
        for block in self.blocks:
            if  hasattr(block, "debug"):
                block.debug(value)

    #---------------------------------------------------------------------------

    def debug(self, value=True, beta=None):
        for block in self.blocks:
            if  hasattr(block, "debug"):
                block.debug(value, beta)


    #---------------------------------------------------------------------------

    def set_dropout(self, p=0.):
        """
        Set dropout rates of DropoutXd to value p. It may be float or list of floats
        """
        layers = get_model_layers(self, kind=(nn.Dropout1d, nn.Dropout2d, nn.Dropout3d))

        if type(p) in (int, float):
            p = [p]

        for i,layer in enumerate(layers):
            layer.p = p[ min(i, len(p)-1) ]

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
    def default():
        return copy.deepcopy(Config(
            input  = 3,
            blocks = "",
            bias   = False,
            norm   = 1,
            drop_d = 2,
            fun    = 'relu',
            res    = 1,
            gamma  = 0.,
            avg    = True,
            flat   = True,
        ))

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
            input    = 3,
            blocks   = "(c64_7_2 n f  m3_2) 2r r128_3_2 r r256_3_2 r r512_3_2 r",
            norm     = 1,
            fun      = 'relu',
            res      = 1,            
        )
        return cfg

    #---------------------------------------------------------------------------

    @staticmethod
    def unit_test():
        B,C,H,W = 1, 1, 28,28
        cnn = CNN(input = C, blocks = "(c64_7_2 n f m3_2) r64 m r128 m")
        #cnn = CNN(CNN.resnet18())
        state = ModelState(cnn)
        state.layers(2, input_size=(1,C,H,W))
        x = torch.rand(B,C,H,W)
        cnn.debug(True)
        y = cnn(x)
        y.mean().backward()
        cnn.update()
        print( cnn.plot(info=True) )
        print(f"ok CNN, output = {tuple(y.shape)}")

        return True

    #---------------------------------------------------------------------------

    def plot(self, w=12, h=3, eps=1e-8, bar_width = 0.75, info=False):
        blocks = self.blocks
        idx = np.arange(len(blocks))

        fig, ax = plt.subplots(1,1, figsize=(w, h))

        ax.grid(ls=":")
        ax.set_xticks(idx)
        ax.set_ylabel("dx/x");  ax.set_xlabel("blocks");

        plt.text(0,0,f" std\n\n", ha='left', transform = ax.transAxes, fontsize=6, family="monospace")

        idx_res, res =  [], []
        for i,block in enumerate(blocks):
            if hasattr(block, 'sqr_dx') and block.sqr_dx is not None and block.sqr_x is not None:
                dx = (block.sqr_dx / (block.sqr_x+eps)).sqrt().cpu().item()
                idx_res.append( i )
                res.append( dx )
        if res:
            ax.set_ylim(bottom=0, top=1.1*np.max(res))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.3f'))
            ax.bar(idx_res, res, width = bar_width, edgecolor ='grey', alpha=0.5)

        names = [''] * len(blocks)
        ymin, ymax = ax.get_ylim()
        for i,block in enumerate(blocks):
            if hasattr(block, 'name'):
                names[i] = block.name
                st = f"{block.std.cpu().item():.1f}" if hasattr(block, 'std') else ""
                plt.text(i, ymin, f"{st}\n{block.name}\n", ha='center', fontsize=6, family="monospace")

        ax2 = ax.twinx()
        idx_gamma_d, gamma_d = [], []
        for i,block in enumerate(blocks):
            if hasattr(block, 'gamma_d') and block.gamma_d is not None:
                idx_gamma_d.append(i)
                gamma_d.append( block.gamma_d.sqrt().cpu().item() )
        if gamma_d:
            ax2.set_ylim(bottom=0, top=1.1*np.max(gamma_d))
            ax2.plot(idx_gamma_d, gamma_d, marker=".")
        ax2.set_ylabel("gamma")

        ax3 = ax.twinx()
        ax3.set_yscale('log')
        idx_gamma_g, gamma_g = [], []
        for i,block in enumerate(blocks):
            if hasattr(block, 'gamma_g') and block.gamma_g is not None:
                idx_gamma_g.append(i)
                gamma_g.append( block.gamma_g.sqrt().cpu().item() )
        if gamma_g:
            ax3.plot(idx_gamma_g, gamma_g, ":", marker=".")
        ax3.set_ylabel("gamma grad")
        ax3.spines["right"].set_position(("outward", 50))

        plt.show()

        if info:
            return {'idx': idx, 'names': names, 'res': res, 'gamma_d': gamma_d, 'gamma_g': gamma_g}
