import copy
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn

from ..config import Config
from ..modelstate import ModelState
from .total  import Create
#========================================================================================

class CNN(nn.Module):
    def __init__(self,  *args, **kvargs) -> None:
        """
        Simple convolutional network: (B,C,H,W) ->  (B,C',H',W')

        The "channel" list specifies the number of channels at the output of each convolutional layer.        
        For example input=(3,32,32) and channel=[8,16] will create two CNN layers and the output of the module will be 16 channels.
        The channel output size (C',H',W') is in cfg.output after the module is created.
        The remaining parameters are specified either as lists (for each layer) or as numbers (then they will be the same in each layer).

        Args
        ------------
            input  (None):
                input tensor shape: (channels, height, width); only channels can be set (channels, None, None)
            channel (list of ints):
                number of channels in each layer
            kernel  (int=3 or list ints):
                size of the convolutional kernel
            stride   (int=1 or list of ints):
                stride of the convolutional kernel
            padding  (int=0 or list of ints):
                padding around the image
            mode     (str='zeros' or list)
                kind of padding ('zeros', 'reflect', 'replicate' or 'circular')
            bias     (bool=False ot list)
                bias in convolution layers
            norm (int=0 or list):
                0: no, 1: BatchNorm2d, 2: InstanceNorm2d, for each layers after Conv2D: 0
            pool_ker (int=0 or list of ints):
                kernel of max-pooling layer
            pool_str (int=0 or list of ints):
                stride of max-pooling kernel
            drop   (int or list of ints)
                1: Dropout, 2: Dropout2d
            fun (str='relu'):
                activation function: gelu, relu, sigmoid, tanh, relu6, swish, hswish, hsigmoid

        Example:
        ------------
        ```
        B,C,H,W = 1, 3, 64,32
        cnn = CNN(input=(C,H,W), channel=[5,7], pool_ker=[0,2])
        x = torch.rand(B,C,H,W)
        y = cnn(x)
        ```
        """
        super().__init__()
        self.cfg = CNN.default()
        self.cfg.set(*args, **kvargs)
        self.create()

    #---------------------------------------------------------------------------

    @staticmethod
    def default():
        return copy.deepcopy(Config(
            input    = None,       # input tensor shape:  (channels, height, width)
            output   = None,       # output tensor shape: (channels, height, width);  sets in create()
            channel  = None,       # number of channels in each layer
            kernel   = 3,          # int or list: size of the convolutional kernel
            stride   = 1,          # int or list: stride of the convolutional kernel
            padding  = 0,          # int or list: padding around the image            
            mode     = 'zeros',    # kind of padding ('zeros', 'reflect', 'replicate' or 'circular')
            bias     = False,      # bias in convolution layers
            norm     = 0,          # 0: no, 1: BatchNorm2d, 2: InstanceNorm2d, for each layers after Conv2D
            pool_ker = 0,          # int or list: max-pooling kernel
            pool_str = 0,          # int or list: stride of max-pooling kernel (if 0, then =pool_ker)
            pool_pad = 0,          # int or list: padding of max-pooling kernel            
            drop     = 2,          # int or list: 1: Dropout, 2: Dropout2d
            fun      = 'relu',     # activation function: gelu, relu, sigmoid, tanh, relu6, swish, hswish, hsigmoid
        ))

    #---------------------------------------------------------------------------

    def forward(self, x):
        x = self.layers(x)
        return x

    #---------------------------------------------------------------------------

    def prepare(self):
        cfg = self.cfg
        assert type(cfg.input) in (tuple,list) and len(cfg.input), "CNN: You must define input shape of image (C,H,W)"
        if type(cfg.channel) == int:
            cfg.channel = [cfg.channel]
        assert type(cfg.channel) in (tuple, list) and len(cfg.channel),  f"CNN: You must define channel (int or list of int): {cfg.channel}"

        n = len(cfg.channel)
        if type(cfg.drop)      == int:    cfg.drop      = [cfg.drop]      * n
        if type(cfg.kernel)    == int:    cfg.kernel    = [cfg.kernel]    * n
        if type(cfg.stride)    == int:    cfg.stride    = [cfg.stride]    * n
        if type(cfg.padding)   == int:    cfg.padding   = [cfg.padding]   * n
        if type(cfg.mode)      == str:    cfg.mode      = [cfg.mode]      * n
        if type(cfg.bias) in (bool,int):  cfg.bias      = [cfg.bias]      * n
        if type(cfg.pool_ker)  == int:    cfg.pool_ker  = [cfg.pool_ker]  * n
        if type(cfg.pool_str)  == int:    cfg.pool_str  = [cfg.pool_str]  * n
        if type(cfg.pool_pad)  == int:    cfg.pool_pad  = [cfg.pool_pad]  * n
        if type(cfg.norm)      == int:    cfg.norm      = [cfg.norm]      * n
        if type(cfg.drop_d)    == int:    cfg.drop_d    = [cfg.drop_d]    * n
        if type(cfg.drop) in (float,int): cfg.drop      = [cfg.drop]      * n


    #---------------------------------------------------------------------------

    def create(self):
        self.prepare()
        cfg = self.cfg
        c, h, w  =  cfg.input
        channels = [ c ] + cfg.channel
        layers = []
        for i in range(len(channels)-1):
            ker, stride, pad   = cfg.kernel[i], cfg.stride[i], cfg.padding[i]
            pool_ker, pool_str, pool_pad = cfg.pool_ker[i], cfg.pool_str[i], cfg.pool_pad[i]
            pool_str = pool_str if pool_str else pool_ker

            layers +=  [ nn.Conv2d(channels[i], channels[i+1], 
                                   kernel_size=ker, stride=stride, padding=pad,
                                   padding_mode=cfg.mode[i], bias=cfg.bias[i]) ]
            if  cfg.norm[i] == 1:
                layers += [ nn.BatchNorm2d( channels[i+1] ) ]
            elif cfg.norm[i] == 2:
                layers += [ nn.InstanceNorm2d( channels[i+1] ) ]

            layers +=  [ Create.activation(self.cfg.fun) ]

            if h: h = int( (h + 2*pad - ker) / stride + 1)
            if w: w = int( (w + 2*pad - ker) / stride + 1)

            if pool_ker > 1:
                layers += [ nn.MaxPool2d(kernel_size=pool_ker, stride=pool_str, padding=pool_pad) ]
                if h: h = int( (h - pool_ker) / pool_str + 1)
                if w: w = int( (w - pool_ker) / pool_str + 1)

            if cfg.drop_d[i] == 1:
                layers += [ nn.Dropout(p=cfg.drop[i]) ]
            elif cfg.drop_d[i] == 2:
                layers += [ nn.Dropout2d(p=cfg.drop[i]) ]

        cfg.output =  (channels[-1], h, w)
        self.layers =  nn.Sequential(*layers)

    #---------------------------------------------------------------------------
    
    @staticmethod
    def unit_test():
        B,C,H,W = 1, 3, 64,32
        cnn = CNN(input=(C,H,W), channel=[5, 7])
        x = torch.rand(B,C,H,W)
        y = cnn(x)
        r1 = y.shape[1:] == cnn.cfg.output
        if not r1:
            print(f"!! CNN: y.shape={y.shape[1:]} != cfg.output={cnn.cfg.output}")

        cnn = CNN(input=(1,None,None), channel=[3, 7], norm=1)
        x = torch.rand(B,1,H,W)
        y = cnn(x)

        if r1:
            print(f"ok CNN, output = {cnn.cfg.output}")

        cnn = CNN(
            input   = (1,28,28),
            channel  =[16,32,64,128],
            kernel  = [8,3,2,4],
            pool_ker= [2,2,0,0],
            padding = [1,1,3,0],
            drop    = [0.3, 0.3, 0.4, 0.5],
            norm    = [1,1,0,0]
        )

        B,C,H,W = 1, 1, 27,28
        x = torch.rand(B,C,H,W)
        y = cnn(x)
        r2 = y.shape[1:] == cnn.cfg.output
        print(y.shape, cnn.cfg.output)
        if not r2:
            print(f"!! CNN: y.shape={y.shape[1:]} != cfg.output={cnn.cfg.output}")


        return r1 and r2

#===============================================================================
#                                    Resnet
#===============================================================================

class ResBlockCNN(nn.Module):
    def __init__(self, *args, **kvargs):
        """
        See ResCNN
        """
        super().__init__()
        self.cfg = ResBlockCNN.default()
        self.cfg.set(*args, **kvargs)
        self.create()        
        self.beta = 0.9

    #---------------------------------------------------------------------------

    @staticmethod
    def default():
        return copy.deepcopy(Config(
            in_channels  = None,   # input tensor shape:: (channels, height, width)
            out_channels = None,   # output tensor shape;  sets in create()
            n_layers = 2,          # number of Conv2D layers in each ResBlockCNN block
            skip     = 1,          # kind of skip-connection in each ResBlockCNN block = 0,1,2
            kernel   = 3,          # size of the convolutional kernel
            stride   = 1,          # stride of the convolutional kernel
            padding  = 0,          # padding around the image
            mode     = 'zeros',    # kind of padding
            norm     = 1,          # 0: none, 1: BatchNorm2d 2: InstanceNorm2d  for each layers
            drop     = 0.0,        # dropout after output
            drop_d   = 1,          # 1: Dropout, 2: Dropout2d
            bias     = False,            
            fun_after= False,
            fun      = 'relu',     # activation function: gelu, relu, sigmoid, tanh, relu6, swish, hswish, hsigmoid
        ))

    #---------------------------------------------------------------------------

    def forward(self, x):                 
        if self.align is  None:
            x = self.block(x)
        else:
            if self.debug:       
                dx = self.block(x)   
                x  = self.align(x)                 

                v_x  = torch.sqrt(torch.square(x. detach()).sum(-1).mean())
                v_dx = torch.sqrt(torch.square(dx.detach()).sum(-1).mean())
                b, b1 = self.beta, 1-self.beta
                if self.avr_x  is None: self.avr_x  = v_x
                else:                   self.avr_x  = b * self.avr_x  + b1 * v_x
                if self.avr_dx is None: self.avr_dx = v_dx            
                else:                   self.avr_dx = b * self.avr_dx + b1 * v_dx

                if self.training and self.std > 0:
                    x = x + dx * (1+torch.randn((x.size(0),1,1,1), device=x.device)*self.std)
                else:
                    x = x + dx 

            else:
                if self.training and self.std > 0:
                    x =  self.align(x)  + self.block(x) * (1+torch.randn((x.size(0),1,1,1), device=x.device)*self.std)
                else:
                    x =  self.align(x)  + self.block(x)   
    
        if self.out_fun is not None:
            x = self.out_fun(x)            
        
        return self.drop(x)

    #---------------------------------------------------------------------------

    def create(self):
        cfg = self.cfg
        layers = []
        channels = [cfg.in_channels] + [cfg.out_channels]*cfg.n_layers
        assert cfg.kernel % 2 == 1, f"kernel in ResBlockCNN should be odd, but got {cfg.kernel}"
        padding  = cfg.kernel // 2
        for i in range(len(channels)-1):
            layers += [ nn.Conv2d(in_channels  = channels[i],
                                  out_channels = channels[i+1],
                                  kernel_size  = cfg.kernel,
                                  padding = padding, padding_mode = cfg.mode,
                                  stride = cfg.stride if i==0 else 1,
                                  bias=cfg.bias) ]

            if  cfg.norm == 1:
                layers += [ nn.BatchNorm2d( channels[i+1] ) ]
            elif cfg.norm == 2:
                layers += [ nn.InstanceNorm2d( channels[i+1] ) ]

            if i < len(channels)-2:
                layers += [ Create.activation(cfg.fun)  ]

        self.block = nn.Sequential(*layers)
        
        assert cfg.skip in [0,1,2], "Error: wrong kind of residual !!!"        
        if cfg.skip == 2 or (cfg.skip == 1 and cfg.in_channels != cfg.out_channels): # надо выравнивать по любому!            
            cfg.skip = 2
            if   cfg.norm == 2:
                norm =  nn.InstanceNorm2d( cfg.out_channels )
            elif cfg.norm == 1:
                norm =  nn.BatchNorm2d( cfg.out_channels )
            else:
                norm = nn.Identity()
            self.align = nn.Sequential(
                nn.Conv2d(cfg.in_channels, cfg.out_channels,
                                   kernel_size=1,  stride=cfg.stride, bias=cfg.bias),
                norm)
            
        elif cfg.skip == 1:
            self.align = nn.Identity()
        elif  cfg.skip == 0:
            self.align = None
        
        self.out_fun = None
        if cfg.fun_after:
            self.out_fun = Create.activation(cfg.fun)   # after skip connection        

        if cfg.drop_d == 2:
            self.drop = nn.Dropout2d(p=cfg.drop)
        else:
            self.drop = nn.Dropout(p=cfg.drop)

        self.debug  = False
        self.avr_x  = None
        self.avr_dx = None
        self.grads  = []
        self.datas  = []        
        self.register_buffer("std",  torch.tensor(float(0.0)))    

    #---------------------------------------------------------------------------

    def update(self):
        i = 0
        for layer in self.block:
            if type(layer) == nn.Conv2d:             
                if layer.weight.grad is  None: 
                    g = 0               
                else:
                    g = layer.weight.grad.square().mean().sqrt() 
                                             
                if len(self.grads) == i:
                    self.grads.append(g)                    
                else:
                    self.grads[i] = self.beta * self.grads[i]  + (1-self.beta) * g

                d = layer.weight.detach().square().mean().sqrt()
                if len(self.datas) == i:                    
                    self.datas.append(d)                    
                else:
                    self.datas[i] = self.beta * self.datas[i]  + (1-self.beta) * d

                i += 1

#===============================================================================

class ResCNN_Old(nn.Module):
    def __init__(self,  *args, **kvargs) -> None:
        """
        ResCNN consists of a sequence of ResBlockCNN blocks
        Each block consists of the same type of Conv2D layers (usually two).
        Image size does not change after ResBlockCNN (only the number of channels can change)
        The first layer has in_channels of input channels and out_channels of output channels.
        The second and following layers have an equal number of input and output channels out_channels

        A skip-connection bypasses the block's Conv2D layer. They can be of the following types:
            * skip = 0: no skip-connection (rare)
            * skip = 1: x + B(x): simple sum of input to block with output from last covolution (required that in_channels==out_channels)
            * skip = 2: A(x)+B(x): the input is passed through Conv2D with a single core to equalize the number of input and output channels (in this case, you can in_channels != out_channels)

        Args
        ------------
            input     (tuple=(None,None,None):
                input tensor shape:  (channels, height, width)
            channel   (int=None or list)
                number of channels in output of i-th  ResBlockCNN block
            layer      (int=2 or list):
                number of Conv2D layers in each ResBlockCNN block
            skip       (int=1 in [0,1,2] or list):
                kind of skip-connection in each ResBlockCNN block; if channel[i-1] != channel[i], skip = 2!
            kernel     (int=3 or list):
                size of the convolutional kernel
            stride     (int=1):
                stride of the convolutional kernel for first layer in block
            padding    (int=0 or list)
                padding around the image
            mode       (str='zeros' or list)
                kind of padding ('zeros', 'reflect', 'replicate' or 'circular')

            norm (int=0 or list):
                0: no, 1: BatchNorm2d, 2: InstanceNorm2d, for each layers after Conv2D: 0
            drop       (float=0.0 or list):
                dropout rate after each block
            drop_d       (int=1):
                dropout dim after each block  1:Dropout, 2: Dropout2d

            pool_ker   (int=0, or list):
                max-pooling kernel
            pool_str   (int=0 or list):
                stride of max-pooling kernel (if 0, then = pool_ker)

            bias       (bool=False or list):
                bias in convolution layers
            fun (str='relu'):
                activation function: gelu, relu, sigmoid, tanh, relu6, swish, hswish, hsigmoid
            averpool   (bool=False)

        Example
        ------------
        B,C,H,W = 1, 3, 64,32
        cnn = ResCNN(   input=(C,H,W),       # будет по 2 слоя в блоке
                        channel=[8, 8, 16])  # где можно skip=1, иначе skip=2
        x = torch.rand(B,C,H,W)
        y = cnn(x)
        """
        super().__init__()
        self.cfg = ResCNN_Old.default()
        self.cfg.set(*args, **kvargs)
        self.create()

    #---------------------------------------------------------------------------

    @staticmethod
    def default():
        return copy.deepcopy(Config(
            input    = None,       # input tensor shape:  (channels, height, width)
            output   = None,       # output tensor shape: (channels, height, width); sets in create()
            channel  = None,       # number of channels in output of i-th  ResBlockCNN block (int or list)
            layer    = 2,          # number of Conv2D layers in each ResBlockCNN block  (int or list)
            skip     = 1,          # kind of skip-connection in each ResBlockCNN block  (int or list) = 0,1,2
            kernel   = 3,          # int or list: size of the convolutional kernel
            stride   = 1,          # int or list: stride of the convolutional kernel for first layer in block
            padding  = 0,          # int or list: padding around the image
            mode     = 'zeros',    # kind of padding ('zeros', 'reflect', 'replicate' or 'circular')
            norm     = 1,          # int or list: BatchNorm2d for each layers
            pool_ker = 0,          # int or list: max-pooling kernel
            pool_str = 0,          # int or list: stride of max-pooling kernel (if 0, then =pool_ker)
            pool_pad = 0,          # int or list: padding of max-pooling kernel
            drop     = 0,          # int or list: dropout after each layer
            drop_d   = 0,          # dropout dim after each block  1:Dropout, 2: Dropout2d
            bias     = False,      # bias in convolution layers
            fun_after=0, 
            fun      = 'relu',     # activation function: gelu, relu, sigmoid, tanh, relu6, swish, hswish, hsigmoid            
            averpool = False,
        ))

    #---------------------------------------------------------------------------

    def forward(self, x):
        return self.layers(x)

    #---------------------------------------------------------------------------

    def set_lists(self):
        cfg = self.cfg
        n = len(cfg.channel)
        if type(cfg.layer)   == int:     cfg.layer   = [cfg.layer]   * n
        if type(cfg.stride)  == int:     cfg.stride  = [cfg.stride]  * n
        if type(cfg.kernel)  == int:     cfg.kernel  = [cfg.kernel]  * n
        if type(cfg.pool_ker)== int:     cfg.pool_ker= [cfg.pool_ker]* n
        if type(cfg.pool_str)== int:     cfg.pool_str= [cfg.pool_str]* n
        if type(cfg.pool_pad)== int:     cfg.pool_pad= [cfg.pool_pad]* n
        if type(cfg.skip)    == int:     cfg.skip    = [cfg.skip]    * n
        if type(cfg.drop) in (float,int):cfg.drop    = [cfg.drop]    * n
        if type(cfg.drop_d)  == int:     cfg.drop_d  = [cfg.drop_d]  * n
        if type(cfg.mode)    == str:     cfg.mode    = [cfg.mode]    * n
        if type(cfg.norm)    == int:     cfg.norm    = [cfg.norm]    * n
        if type(cfg.bias) in (bool,int): cfg.bias    = [cfg.bias]    * n
        if type(cfg.fun_after) in (bool,int): cfg.fun_after = [cfg.fun_after]*n

    #---------------------------------------------------------------------------

    def create(self):
        cfg = self.cfg
        assert type(cfg.input)   in (tuple,list) and len(cfg.input) == 3, f"got {cfg.input}"
        assert type(cfg.channel) in (tuple,list) and len(cfg.channel),    f"got {cfg.channel}"
        self.set_lists()

        channels = [cfg.input[0]] + cfg.channel
        self.layers = []
        for i in range(len(channels)-1):
            self.layers +=  [ 
                ResBlockCNN(in_channels=channels[i],
                            out_channels=channels[i+1],
                            kernel=cfg.kernel[i],
                            skip=cfg.skip[i], n_layers=cfg.layer[i],
                            stride=cfg.stride[i],
                            mode=cfg.mode[i], norm=cfg.norm[i], bias=cfg.bias[i],
                            drop=cfg.drop[i], drop_d = cfg.drop_d[i], fun_after=cfg.fun_after[i]
                            )]
            if cfg.pool_ker[i] > 1:
                pool_str = cfg.pool_str[i] if cfg.pool_str[i] > 0 else cfg.pool_ker[i]                
                self.layers += [ nn.MaxPool2d(kernel_size = cfg.pool_ker[i],
                                              stride      = pool_str, padding=cfg.pool_pad[i]) ]

        if cfg.averpool:
            self.layers += [ nn.AdaptiveAvgPool2d((1, 1)) ]

        self.layers = nn.Sequential(*self.layers)
        cfg.output = self.calc_output(False)

    #---------------------------------------------------------------------------

    def calc_output(self, set_lst=True):
        """
        Calculates the output shape for the given cfg
        """
        cfg = self.cfg
        if set_lst:
            self.set_lst()

        h, w = cfg.input[1:]
        channels = [cfg.input[0]] + cfg.channel
        for i in range(len(channels)-1):
            if cfg.stride[i] > 1:
                if h: h = int( (h + 2*(cfg.kernel[i]//2) - cfg.kernel[i]) / cfg.stride[i] + 1)
                if w: w = int( (h + 2*(cfg.kernel[i]//2) - cfg.kernel[i]) / cfg.stride[i] + 1)
            if cfg.pool_ker[i] > 1:
                if h: h = int( (h - cfg.pool_ker[i]) / cfg.pool_ker[i] + 1)
                if w: w = int( (w - cfg.pool_ker[i]) / cfg.pool_ker[i] + 1)
        if cfg.averpool:
            h, w = 1, 1

        return (channels[-1], h, w)

    #---------------------------------------------------------------------------

    def std(self, stds, i=None):
        """
        Set std value (agumentation) for all blocks or i-th

        Example
        ------------
        ```
            cnn.std( 0.2 )              # equal value for all block 
            cnn.std( [0.2, 0.1, 0.1] )  # some  value for each block 
            cnn.std( 0.2, 5)            # for 5-th block (from 0)
        ```        
        """
        n_blocks = len(self.cfg.channel)

        if i is not None:
            assert type(stds) in [float, int] and i >=0 and i < n_blocks, f"Wrong stds={stds} for i={i}"                
            for layer in self.layers:
                if type(layer) is ResBlockCNN:
                    if i <= 0:
                        layer.std.fill_(float(stds))
                        return
                    i -= 1                                

        if type(stds) in [float, int]:
            stds = [stds] * n_blocks

        assert type(stds) in [list, tuple] and len(stds) == n_blocks, f"Wrong stds={stds}"
        i = 0
        for layer in self.layers:
            if type(layer) is ResBlockCNN:
                layer.std.fill_(float(stds[i]))
                i += 1


    #---------------------------------------------------------------------------

    def debug(self, value=True, beta=None):
        for layer in self.layers:
            if type(layer) is ResBlockCNN:
                layer.debug = value
                if not value:
                    layer.avr_x = layer.avr_dx  = None        
                    layer.grads  = []
                if beta is not None:
                    layer.beta = beta

    #---------------------------------------------------------------------------

    def update(self):
        for layer in self.layers:
            if type(layer) is ResBlockCNN:
                layer.update()

    #---------------------------------------------------------------------------

    def plot(self, w=12, h=3, eps=1e-8):
        fig, ax = plt.subplots(1,1, figsize=(w, h))

        plt.text(0,0,f" res\n std\n", ha='left', transform = ax.transAxes, fontsize=8)
        weights, dx, i, ma = [], [], 0, 0
        for block in self.layers:
            if type(block) is ResBlockCNN:                
                ww = [d.cpu().item() for d in block.datas]
                weights.append(ww)
                ma = max(ma, len(ww))
                if block.avr_dx is not None:
                    dx.append( (block.avr_dx/(block.avr_x+eps)).cpu().item() )

                plt.text(i,0,f"{block.cfg.skip}\n{block.std.item():.2f}\n", ha='center', fontsize=8)
                i += 1
        weights = [ ww + [0]*(ma-len(ww)) for ww in weights]
        
        idxs = np.arange( len(self.cfg.channel) )        
        ax.set_xticks(idxs)
        if len(dx):
            ax.bar(idxs, dx, alpha=0.8, color="lightgray", ec="black")
            ax.set_ylim(0, np.max(np.array(dx).flatten())*1.1) 
        ax.set_ylabel("dx/x");  ax.set_xlabel("blocks");
        ax.grid(ls=":")

        ax2 = ax.twinx() 
        weights = np.array(weights).transpose()        
        for i,w in enumerate(weights):            
            ax2.plot(idxs, w, marker=".", label=f'{i}')
        ax2.set_ylabel("|weight|")            
        ax2.set_ylim(0, weights.flatten().max()*1.1) 
        ax2.legend(loc='upper right')

        grads, ma = [], 0
        for block in self.layers:
            if type(block) is ResBlockCNN:                
                grads.append([ g.cpu().item() for g in block.grads])
                ma = max(ma, len(block.grads))

        if len (grads):            
            grads = [ g + [0]*(ma-len(g)) for g in grads]
            grads = np.array(grads).transpose()                                    
            ax3 = ax.twinx() 
            for i,g in enumerate(grads):            
                ax3.plot(idxs, g, ls=":", marker=".")
            ax3.spines["right"].set_position(("outward", 50))
            ax3.set_ylim(0, grads.flatten().max()*1.1) 
            ax3.set_ylabel("--- |grad|")            
         

        plt.show()
        
    #---------------------------------------------------------------------------

    @staticmethod
    def resnet18():
        """
        Equal:
        from torchvision.models import resnet18
        model = resnet18()
        """
        cfg = ResCNN_Old.default()
        cfg(
            input    = (3, 100, 100),
            channel  = [64,  64,  64, 128, 128, 256, 256, 512, 512],             
            stride   = [ 2,   1,   1,   2,   1,   2,   1,   2,   1],
            skip     = [ 0] + [1]*8,
            layer    = [ 1] + [2]*8,
            kernel   = [ 7] + [3]*8,            
            pool_ker = [ 3] + [0]*8,
            pool_str = [ 2] + [0]*8,
            pool_pad = [ 1] + [0]*8,
            fun_after= [ 1] + [0]*8,
            norm     = 1,            
            averpool = True,
        )        
        return cfg

    #---------------------------------------------------------------------------

    @staticmethod
    def unit_test():
        B,C,H,W = 1, 3, 64,32
        cnn = ResCNN_Old(   input=(C,H,W),
                        channel=[8, 8, 16],
                        skip  = [2, 1, 2] )
        x = torch.rand(B,C,H,W)
        y = cnn(x)
        y.square().mean().backward()
        cnn.update()        
        cnn.plot()

        r1 = y.shape[1:] == cnn.cfg.output
        if not r1:
            print(f"!! ResCNN: y.shape={y.shape[1:]} != cfg.output={cnn.cfg.output}")
        if r1:
            print(f"ok ResCNN, output = {cnn.cfg.output}")

        return r1


#===============================================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    From: https://github.com/osmr/imgclsmob/blob/68335927ba27f2356093b985bada0bc3989836b1/pytorch/pytorchcv/models/common.py#L731

    Args
    ----------
    n_channels : int
        Number of channels.
    reduction : int, default 16
        Squeeze reduction value.
    approx_sigmoid : bool, default False
        Whether to use approximated sigmoid function.
    activation : function, or str, or nn.Module
        Activation function or name of activation function.
    """
    def __init__(self,
                 n_channels,
                 reduction=16,
                 approx_sigmoid=False,
                 activation=(lambda: nn.ReLU(inplace=True))):
        super(SEBlock, self).__init__()
        mid_cannels = n_channels // reduction

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 =  nn.Conv2d(in_channels=n_channels, out_channels=mid_cannels, kernel_size=1, stride=1, bias=True)
        self.activ = Create.activation(activation)
        self.conv2 = nn.Conv2d(in_channels=mid_cannels, out_channels=n_channels, kernel_size=1, stride=1, bias=True)
        self.sigmoid = Create.activation("hsigmoid") if approx_sigmoid else nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        x = x * w
        return x

#===============================================================================
