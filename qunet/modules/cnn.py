import copy
import torch, torch.nn as nn

from ..utils   import Config
#========================================================================================

class CNN(nn.Module):
    def __init__(self,  *args, **kvargs) -> None:
        """
        Simple convolutional network: (B,C,H,W) ->  (B,C',H',W')

        The number of layers is set by the channel parameter. This is the number of channels at the output of each layer.
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
            norm (int=0 or list):
                0: no, 1: BatchNorm2d, 2: InstanceNorm2d, for each layers after Conv2D: 0
            pool_ker (int=0 or list of ints):
                max-pooling kernel
            pool_str (int=0 or list of ints):
                stride of max-pooling kernel
            drop     (float=0.0 or list of floats):
                dropout after each layer
            drop_d   (int or list of ints)
                1: Dropout, 2: Dropout2d

        Example:
        ------------
        ```
        B,C,H,W = 1, 3, 64,32
        cnn = CNN(input=(C,H,W), channel=[5, 7], pool_ker=[0,2])
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
            norm     = 0,          # 0: no, 1: BatchNorm2d, 2: InstanceNorm2d, for each layers after Conv2D
            padding  = 0,          # int or list: padding around the image
            pool_ker = 0,          # int or list: max-pooling kernel
            pool_str = 0,          # int or list: stride of max-pooling kernel (if 0, then =pool_ker)
            drop     = 0,          # int or list: dropout after each layer (ReLU)
            drop_d   = 1,          # int or list: 1: Dropout, 2: Dropout2d
        ))

    #---------------------------------------------------------------------------

    def forward(self, x):
        x = self.layers(x)
        return x

    #---------------------------------------------------------------------------

    def prepare(self):
        cfg = self.cfg
        assert type(cfg.input) == tuple, "CNN: You must define input shape of image (C,H,W)"
        if type(cfg.channel) == int:
            cfg.channel = [cfg.channel]
        assert type(cfg.channel) == list, f"CNN: You must define channel (int or list of int): {cfg.channel}"

        n = len(cfg.channel)
        if type(cfg.drop)      == int:    cfg.drop      = [cfg.drop]      * n
        if type(cfg.kernel)    == int:    cfg.kernel    = [cfg.kernel]    * n
        if type(cfg.stride)    == int:    cfg.stride    = [cfg.stride]    * n
        if type(cfg.padding)   == int:    cfg.padding   = [cfg.padding]   * n
        if type(cfg.pool_ker)  == int:    cfg.pool_ker  = [cfg.pool_ker]  * n
        if type(cfg.pool_str)  == int:    cfg.pool_str  = [cfg.pool_str]  * n
        if type(cfg.norm)      == int:    cfg.norm      = [cfg.norm]      * n
        if type(cfg.drop_d)    == int:    cfg.drop_d    = [cfg.drop_d]    * n
        if type(cfg.drop) in (float,int): cfg.drop      = [cfg.drop]      * n


    #---------------------------------------------------------------------------

    def create(self):
        self.prepare()
        cfg = self.cfg
        c, w, h  =  cfg.input
        channels = [ c ] + cfg.channel
        layers = []
        for i in range(len(channels)-1):
            kernel, stride     = cfg.kernel  [i], cfg.stride[i]
            padding            = cfg.padding [i]
            pool_ker, pool_str = cfg.pool_ker[i], cfg.pool_str[i]
            pool_str = pool_str if pool_str else pool_ker

            layers +=  [ nn.Conv2d(channels[i], channels[i+1], kernel_size=kernel, stride=stride, padding=padding) ]
            if  cfg.norm[i] == 1:
                layers += [ nn.BatchNorm2d( channels[i+1] ) ]
            elif cfg.norm[i] == 2:
                layers += [ nn.InstanceNorm2d( channels[i+1] ) ]

            layers +=  [ nn.ReLU()]

            if h: h = int( (h + 2*padding - kernel) / stride + 1)
            if w: w = int( (w + 2*padding - kernel) / stride + 1)

            if pool_ker > 1:
                layers += [ nn.MaxPool2d(kernel_size=pool_ker, stride=pool_str) ]
                if h: h = int( (h - pool_ker) / pool_str + 1)
                if w: w = int( (w - pool_ker) / pool_str + 1)

            if cfg.drop_d[i] == 1:
                layers += [ nn.Dropout(p=cfg.drop[i]) ]
            elif cfg.drop_d[i] == 2:
                layers += [ nn.Dropout(p=cfg.drop[i]) ]

        cfg.output =  (channels[-1], w, h)
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
        ResBlockCNN состоит из однотипных Conv2D слоёв (обычно двух).
        Размер изображения после ResBlockCNN не изменяется (может поменяться только число каналов)
        Первый слой имеет in_channels входных каналов и out_channels выходных.
        Второй и последующие слои имеют равное число входных и выходных каналов out_channels

        Слои Conv2D окружены skip-connections.  Они могут быть следующих типов:
            * skip = 0: отсутствуют (редко)
            * skip = 1: простая сумма входа в блок с выходом из последней коволюции (необходимо, чтобы in_channels==out_channels)
            * skip = 2: вход пропускается через Conv2D с единичным ядром для выраванивания числа входных и выходных каналов (в этом случае можно in_channels != out_channels)

        If stride > 1 it's only for first Conv2d
        Always padding = kernel // 2

        Args
        ------------
            in_channels  (int):
            out_channels (int):
            n_layers     (int=2):
            kernel       (int=3):
            stride       (int=1):
            mode         (str='zeros'):
            batchnorm    (bool=True):
            bias         (bool=False)
            skip          (int=0)

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
            skip     = 0,          # kind of skip-connection in each ResBlockCNN block = 0,1,2
            kernel   = 3,          # size of the convolutional kernel
            stride   = 1,          # stride of the convolutional kernel
            padding  = 0,          # padding around the image
            mode      = 'zeros',   # kind of padding
            batchnorm = 0,         # BatchNorm2d for each layers
            drop     = 0,          # dropout after each layer
            bias      = False,
        ))

    #---------------------------------------------------------------------------

    def forward(self, x):
        y = self.block(x)
        if self.align is not None:
            y2 = self.align(x)
            if self.batchnorm is not None:
                y2 = self.batchnorm(y2)
            y += y2

        y = self.relu(y)
        return y

    #---------------------------------------------------------------------------

    def create(self):
        cfg = self.cfg
        layers = []
        channels = [cfg.in_channels] + [cfg.out_channels]*cfg.n_layers
        padding  = cfg.kernel // 2
        for i in range(len(channels)-1):
            layers += [ nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1],
                                  kernel_size=cfg.kernel,
                                  padding=padding, padding_mode=cfg.mode,
                                  stride=cfg.stride if i==0 else 1,  bias=cfg.bias) ]
            if cfg.batchnorm:
                layers += [ nn.BatchNorm2d(channels[i+1]) ]
            if i < len(channels)-2:
                layers += [ nn.ReLU(inplace=True) ]

        self.block = nn.Sequential(*layers)

        assert cfg.skip in [0,1,2], "Error: wrong kind of residual !!!"
        self.batchnorm = None
        if  cfg.skip == 0:
            self.align = None
        elif cfg.skip == 1:
            self.align = nn.Identity()
        elif cfg.skip == 2:                                       # !
            self.align     = nn.Conv2d(cfg.in_channels, cfg.out_channels, kernel_size=1,  stride=cfg.stride, bias=cfg.bias)
            self.batchnorm = nn.BatchNorm2d(cfg.out_channels) if cfg.batchnorm else None

        self.relu = nn.ReLU()   # after skip connection

#===============================================================================

class ResCNN(nn.Module):
    def __init__(self,  *args, **kvargs) -> None:
        """
        ResCNN состоит из последовательности ResBlockCNN блоков
        Каждый блок состит из однотипных Conv2D слоёв (обычно двух).
        Размер изображения после ResBlockCNN не изменяется (может поменяться только число каналов)

        Слои Conv2D блока окружены skip-connections. Они могут быть следующих типов:
            * skip = 0: отсутствуют (редко)
            * skip = 1: простая сумма входа в блок с выходом из последней коволюции (необходимо, чтобы in_channels==out_channels)
            * skip = 2: вход пропускается через Conv2D с единичным ядром для выраванивания числа входных и выходных каналов (в этом случае можно in_channels != out_channels)

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
                kind of padding
            batchnorm  (int=0 or list):
                BatchNorm2d for each layers
            pool_ker   (int=0, or list):
                max-pooling kernel
            pool_str   (int=0 or list):
                stride of max-pooling kernel (if 0, then = pool_ker)
            drop       (float=0.0 or list):
                dropout after each layer
            bias       (bool=False or list):
            averpool   (bool=False)
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
            mode      = 'zeros',   # kind of padding
            batchnorm = 0,         # int or list: BatchNorm2d for each layers
            pool_ker = 0,          # int or list: max-pooling kernel
            pool_str = 0,          # int or list: stride of max-pooling kernel (if 0, then =pool_ker)
            drop     = 0,          # int or list: dropout after each layer
            bias      = False,
            averpool  = False,
        ))

    #---------------------------------------------------------------------------

    def forward(self, x):
        return self.model(x)

    #---------------------------------------------------------------------------

    def set_lists(self):
        cfg = self.cfg
        if type(cfg.layer)   == int:    cfg.layer   = [cfg.layer]   * len(cfg.channel)
        if type(cfg.stride)  == int:    cfg.stride  = [cfg.stride]  * len(cfg.channel)
        if type(cfg.kernel)  == int:    cfg.kernel  = [cfg.kernel]  * len(cfg.channel)
        if type(cfg.pool_ker)== int:    cfg.pool_ker= [cfg.pool_ker]* len(cfg.channel)
        if type(cfg.skip)    == int:    cfg.skip    = [cfg.skip]    * len(cfg.channel)

    #---------------------------------------------------------------------------

    def create(self):
        cfg = self.cfg
        self.set_lists()
        channels = [cfg.input[0]] + cfg.channel
        self.layers = []
        for i in range(len(channels)-1):
            if channels[i] != channels[i+1] and cfg.skip[i] > 0:
                cfg.skip[i] = 2

            self.layers +=  [ ResBlockCNN(in_channels=channels[i], out_channels=channels[i+1], kernel=cfg.kernel[i],
                                          skip=cfg.skip[i], n_layers=cfg.layer[i], stride=cfg.stride[i],
                                          mode=cfg.mode, batchnorm=cfg.batchnorm, bias=cfg.bias)  ]
            if cfg.pool_ker[i] > 1:
                self.layers += [ nn.MaxPool2d(kernel_size = cfg.pool_ker[i], stride = cfg.pool_ker[i]) ]

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

    def unit_test():
        B,C,H,W = 1, 3, 64,32
        cnn = ResCNN(input=(C,H,W), channel=[5, 7])
        x = torch.rand(B,C,H,W)
        y = cnn(x)
        r1 = y.shape[1:] == cnn.cfg.output
        if not r1:
            print(f"!! ResCNN: y.shape={y.shape[1:]} != cfg.output={cnn.cfg.output}")
        if r1:
            print(f"ok ResCNN, output = {cnn.cfg.output}")

        return r1


#===============================================================================