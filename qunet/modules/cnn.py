import copy
import torch, torch.nn as nn

from ..utils import Config
from .total  import get_activation
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
            drop     (float=0.0 or list of floats):
                dropout after each layer
            drop_d   (int or list of ints)
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
            drop     = 0.0,        # int or list: dropout after each layer (ReLU)
            drop_d   = 2,          # int or list: 1: Dropout, 2: Dropout2d
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
            pool_ker, pool_str = cfg.pool_ker[i], cfg.pool_str[i]
            pool_str = pool_str if pool_str else pool_ker

            layers +=  [ nn.Conv2d(channels[i], channels[i+1], 
                                   kernel_size=ker, stride=stride, padding=pad,
                                   padding_mode=cfg.mode[i], bias=cfg.bias[i]) ]
            if  cfg.norm[i] == 1:
                layers += [ nn.BatchNorm2d( channels[i+1] ) ]
            elif cfg.norm[i] == 2:
                layers += [ nn.InstanceNorm2d( channels[i+1] ) ]

            layers +=  [ get_activation(self.cfg.fun) ]

            if h: h = int( (h + 2*pad - ker) / stride + 1)
            if w: w = int( (w + 2*pad - ker) / stride + 1)

            if pool_ker > 1:
                layers += [ nn.MaxPool2d(kernel_size=pool_ker, stride=pool_str) ]
                if h: h = int( (h - pool_ker) / pool_str + 1)
                if w: w = int( (w - pool_ker) / pool_str + 1)

            if cfg.drop_d[i] == 1:
                layers += [ nn.Dropout(p=cfg.drop[i]) ]
            elif cfg.drop_d[i] == 2:
                layers += [ nn.Dropout2d(p=cfg.drop[i]) ]

        cfg.output =  (channels[-1], h, w)
        self.layers =  nn.Sequential(*layers)

    #---------------------------------------------------------------------------

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

class ResBlockCNN(nn.Module):
    def __init__(self, *args, **kvargs):
        """
        ResBlockCNN consists of the Conv2D layers (usually two).
        Image size does not change after ResBlockCNN (only the number of channels)
        The first layer has in_channels of input channels and out_channels of output channels.
        The second and following layers have an equal number of input and output channels out_channels

        Conv2D layers are surrounded by skip-connections. They can be of the following types:
            * skip = 0: none (rare)
            * skip = 1: simple sum of the block entry with the last covolution exit (required that in_channels==out_channels)
            * skip = 2: the input is skipped through Conv2D with a single core to equalize the number of input and output channels (in this case you can in_channels != out_channels)

        If stride > 1 it's only for first Conv2d
        Always padding = kernel // 2

        Args
        ------------
            in_channels  (int):
                input tensor shape: (channels, height, width)
            out_channels (int):
                output tensor shape;  sets in create()
            n_layers     (int=2):
                number of Conv2D layers in each ResBlockCNN block
            kernel       (int=3):
                size of the convolutional kernel
            stride       (int=1):
                stride of the convolutional kernel
            mode         (str='zeros'):
                kind of padding ('zeros', 'reflect', 'replicate' or 'circular')
            norm        (bool=True):
                0: none, 1: BatchNorm2d 2: InstanceNorm2d  for each layers
            bias         (bool=False)
                bias in convolution layers
            skip          (int=1):
                kind of skip-connection in each ResBlockCNN block = 0,1,2
            drop         (float=0.0):
                dropout rate after block
            drop_d       (int=1):
                dropout dim  1:Dropout, 2: Dropout2d
            fun (str='relu'):
                activation function: gelu, relu, sigmoid, tanh, relu6, swish, hswish, hsigmoid

        """
        super().__init__()
        self.cfg = ResBlockCNN.default()
        self.cfg.set(*args, **kvargs)
        self.create()

    #---------------------------------------------------------------------------

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
            norm     = 0,          # 0: none, 1: BatchNorm2d 2: InstanceNorm2d  for each layers
            drop     = 0.0,        # dropout after output
            drop_d   = 1,          # 1: Dropout, 2: Dropout2d
            bias     = False,
            fun      = 'relu',     # activation function: gelu, relu, sigmoid, tanh, relu6, swish, hswish, hsigmoid
        ))

    #---------------------------------------------------------------------------

    def forward(self, x):
        y = self.block(x)
        if self.align is not None:
            y = self.norm( y + self.align(x) )
        y = self.out_fun(y)
        y = self.drop(y)
        return y

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
                layers += [ get_activation(cfg.fun)  ]

        self.block = nn.Sequential(*layers)

        assert cfg.skip in [0,1,2], "Error: wrong kind of residual !!!"
        if cfg.skip == 2 or cfg.in_channels != cfg.out_channels: # надо выравнивать по любому!
            self.align = nn.Conv2d(cfg.in_channels, cfg.out_channels,
                                   kernel_size=1,  stride=cfg.stride, bias=cfg.bias)
        elif cfg.skip == 1:
            self.align = nn.Identity()
        elif  cfg.skip == 0:
            self.align = None

        if   cfg.norm == 2:
            self.norm =  nn.InstanceNorm2d( cfg.out_channels )
        elif cfg.norm == 1:
            self.norm =  nn.BatchNorm2d( cfg.out_channels )
        else:
            self.norm = nn.Identity()

        self.out_fun = get_activation(cfg.fun)   # after skip connection

        if cfg.drop_d == 2:
            self.drop = nn.Dropout2d(p=cfg.drop)
        else:
            self.drop = nn.Dropout(p=cfg.drop)

#===============================================================================

class ResCNN(nn.Module):
    def __init__(self,  *args, **kvargs) -> None:
        """
        ResCNN consists of a sequence of ResBlockCNN blocks
        Each block consists of the same type of Conv2D layers (usually two).
        Image size does not change after ResBlockCNN (only the number of channels can change)

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
                stride of the convolutional kernel
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
        self.cfg = ResCNN.default()
        self.cfg.set(*args, **kvargs)
        self.create()

    #---------------------------------------------------------------------------

    def default():
        return copy.deepcopy(Config(
            input    = None,       # input tensor shape:  (channels, height, width)
            output   = None,       # output tensor shape: (channels, height, width); sets in create()
            channel  = None,       # number of channels in output of i-th  ResBlockCNN block (int or list)
            layer    = 2,          # number of Conv2D layers in each ResBlockCNN block  (int or list)
            skip     = 1,          # kind of skip-connection in each ResBlockCNN block  (int or list) = 0,1,2
            kernel   = 3,          # int or list: size of the convolutional kernel
            stride   = 1,          # int or list: stride of the convolutional kernel
            padding  = 0,          # int or list: padding around the image
            mode     = 'zeros',    # kind of padding ('zeros', 'reflect', 'replicate' or 'circular')
            norm     = 0,          # int or list: BatchNorm2d for each layers
            pool_ker = 0,          # int or list: max-pooling kernel
            pool_str = 0,          # int or list: stride of max-pooling kernel (if 0, then =pool_ker)
            drop     = 0,          # int or list: dropout after each layer
            drop_d   = 0,          # dropout dim after each block  1:Dropout, 2: Dropout2d
            bias     = False,      # bias in convolution layers
            fun      = 'relu',     # activation function: gelu, relu, sigmoid, tanh, relu6, swish, hswish, hsigmoid
            averpool = False,
        ))

    #---------------------------------------------------------------------------

    def forward(self, x):
        return self.model(x)

    #---------------------------------------------------------------------------

    def set_lists(self):
        cfg = self.cfg
        if type(cfg.layer)   == int:     cfg.layer   = [cfg.layer]   * len(cfg.channel)
        if type(cfg.stride)  == int:     cfg.stride  = [cfg.stride]  * len(cfg.channel)
        if type(cfg.kernel)  == int:     cfg.kernel  = [cfg.kernel]  * len(cfg.channel)
        if type(cfg.pool_ker)== int:     cfg.pool_ker= [cfg.pool_ker]* len(cfg.channel)
        if type(cfg.pool_str)== int:     cfg.pool_str= [cfg.pool_str]* len(cfg.channel)
        if type(cfg.skip)    == int:     cfg.skip    = [cfg.skip]    * len(cfg.channel)
        if type(cfg.drop) in (float,int):cfg.drop    = [cfg.drop]    * len(cfg.channel)
        if type(cfg.drop_d)  == int:     cfg.drop_d  = [cfg.drop_d]  * len(cfg.channel)
        if type(cfg.mode)    == str:     cfg.mode    = [cfg.mode]    * len(cfg.channel)
        if type(cfg.norm)    == int:     cfg.norm    = [cfg.norm]    * len(cfg.channel)
        if type(cfg.bias) in (bool,int): cfg.bias    = [cfg.bias]    * len(cfg.channel)

    #---------------------------------------------------------------------------

    def create(self):
        cfg = self.cfg
        assert type(cfg.input)   in (tuple,list) and len(cfg.input) == 3, f"got {cfg.input}"
        assert type(cfg.channel) in (tuple,list) and len(cfg.channel),    f"got {cfg.channel}"
        self.set_lists()

        channels = [cfg.input[0]] + cfg.channel
        self.layers = []
        for i in range(len(channels)-1):
            self.layers +=  [ ResBlockCNN(in_channels=channels[i],
                                          out_channels=channels[i+1],
                                          kernel=cfg.kernel[i],
                                          skip=cfg.skip[i], n_layers=cfg.layer[i],
                                          stride=cfg.stride[i],
                                          mode=cfg.mode[i], norm=cfg.norm[i], bias=cfg.bias[i],
                                          drop=cfg.drop[i], drop_d = cfg.drop_d[i]
                                        )
                            ]
            if cfg.pool_ker[i] > 1:
                pool_str = cfg.pool_str[i] if cfg.pool_str[i] > 0 else cfg.pool_ker[i]
                self.layers += [ nn.MaxPool2d(kernel_size = cfg.pool_ker[i],
                                              stride      = pool_str) ]

        if cfg.averpool:
            self.layers += [ nn.AdaptiveAvgPool2d((1, 1)) ]

        self.model = nn.Sequential(*self.layers)
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

    def get(self, x):
        res = []
        for layer in self.layers:
            x = layer(x)
            res.append(x.data.cpu().numpy())
        return res

    #---------------------------------------------------------------------------

    def resnet18():
        cfg = ResCNN.default()
        cfg(
            channel = [], 
        )
        return cfg

    #---------------------------------------------------------------------------

    def unit_test():
        B,C,H,W = 1, 3, 64,32
        cnn = ResCNN(   input=(C,H,W),
                        channel=[8, 8, 16],
                        skip  = [2, 1, 2] )
        x = torch.rand(B,C,H,W)
        y = cnn(x)

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
        self.activ = get_activation(activation)
        self.conv2 = nn.Conv2d(in_channels=mid_cannels, out_channels=n_channels, kernel_size=1, stride=1, bias=True)
        self.sigmoid = get_activation("hsigmoid") if approx_sigmoid else nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        x = x * w
        return x

#===============================================================================