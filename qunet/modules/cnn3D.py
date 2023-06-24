import copy
import torch, torch.nn as nn

from ..config   import Config

class CNN3DBlock(nn.Module):
    def __init__(self, in_channels,  out_channels, kernel = 3, stride = 1, layers=2,  mode = 'zeros',  batchnorm=True, bias=False, residual=0):
        super(CNN3DBlock, self).__init__()

        layers, channels, padding = [],  [in_channels] + [out_channels]*layers,   kernel // 2
        for i in range(len(channels)-1):
            layers += [ nn.Conv3d(channels[i], channels[i+1], kernel_size=kernel, padding=padding, padding_mode=mode,
                                  stride=stride if i==0 else 1,  bias=bias) ]
            if batchnorm:
                layers += [ nn.BatchNorm3d(num_features=channels[i+1]) ]
            if i < len(channels)-2:
                layers += [ nn.ReLU(inplace=True) ]

        self.block = nn.Sequential(*layers)

        self.batchnorm = None
        if  residual == 0:
            self.align = None
        elif residual == 1:
            self.align = nn.Identity()
        elif residual == 2:
            self.align     = nn.Conv3d(in_channels, out_channels, kernel_size=1,  stride=stride, bias=bias)
            self.batchnorm = nn.BatchNorm3d(num_features=out_channels) if batchnorm else None
        elif residual == 3:
            self.align = nn.Parameter( torch.randn(in_channels, out_channels) )  # TODO
        else:
            print("Error: wrong kind of residual !!!")

        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.block(x)
        if self.align is not None:
            y2 = self.align(x)
            if self.batchnorm is not None:
                y2 = self.batchnorm(y2)
            y += y2

        y = self.relu(y)
        return y

class CNN3D(nn.Module):
    def __init__(self, *args, **kvargs):
        """
        Simple convolutional 3D network: (B,C,H,W,D) ->  (B,C',H',W',D')

        The number of layers is set by the channel parameter. This is the number of channels at the output of each layer.
        For example input=(3,32,32,32) and channel=[8,16] will create two CNN layers and the output of the module will be 16 channels.
        The channel output size (C',H',W',D') is in cfg.output after the module is created.
        The remaining parameters are specified either as lists (for each layer) or as numbers (then they will be the same in each layer).

        Args:
            * `input=None`:   input tensor shape:: (channels, height, width, depth)
            * `outputs=None`: output tensor shape;
            * `layers=2`:     layers in block
            * `channels=[]`:  conv3D chanels
            * `kernels=3`:    int or list: size of the convolutional kernel
            * `strides=1`:    int or list: stride of the convolutional kernel
            * `mode='zeros'`:
            * `batchnorm=True`:  batchnorm
            * `bias=False`:      bias
            * `pools=0.`:        pooling
            * `hiddens=[]`:      add output hidden layers
            * `residual=[]`:     add residual path
            * `averpool=True`:   add average pooling

        Example:
        ```
            cnn = CNN3D(input=(3,32,32,32), channel=[16,32], kernel=3)
        ```
        """

        super().__init__()
        self.cfg = CNN3D.default()
        self.cfg.set(*args, **kvargs)
        self.create()

    def prepare(self):
        cfg = self.cfg
        if type(cfg.layers) == int:      cfg.layers   = [cfg.layers]   * len(cfg.channels)
        if type(cfg.strides)  == int:    cfg.strides  = [cfg.strides]  * len(cfg.channels)
        if type(cfg.pools)    == int:    cfg.pools    = [cfg.pools]    * len(cfg.channels)
        if type(cfg.kernels)  == int:    cfg.kernels  = [cfg.kernels]  * len(cfg.channels)
        if type(cfg.dropouts) == float:  cfg.dropouts = [cfg.dropouts] * (len(cfg.hiddens) + 1)

    def create(self):
        self.prepare()
        cfg = self.cfg
        in_channels, h, w, d = cfg.input
        channels = [in_channels] + cfg.channels
        layers = []
        for i in range(len(channels)-1):
            layers +=  [ CNN3DBlock(channels[i], channels[i+1], kernel=cfg.kernels[i], residual=cfg.residual[i], layers=cfg.layers[i], stride=cfg.strides[i], mode=cfg.mode, batchnorm=cfg.batchnorm, bias=cfg.bias)  ]
            if cfg.strides[i] > 1:
                h = int( (h + 2*(cfg.kernels[i]//2) - cfg.kernels[i]) / cfg.strides[i] + 1)
                w = int( (w + 2*(cfg.kernels[i]//2) - cfg.kernels[i]) / cfg.strides[i] + 1)
                d = int( (d + 2*(cfg.kernels[i]//2) - cfg.kernels[i]) / cfg.strides[i] + 1)
            if cfg.pools[i] > 1:
                layers += [ nn.MaxPool3d(kernel_size = cfg.pools[i], stride = cfg.pools[i]) ]
                h = int( (h - cfg.pools[i]) / cfg.pools[i] + 1)
                w = int( (w - cfg.pools[i]) / cfg.pools[i] + 1)
                d = int( (d - cfg.pools[i]) / cfg.pools[i] + 1)

        if cfg.averpool:
            layers += [ nn.AdaptiveAvgPool3d((1, 1, 1)) ]
            h, w, d = 1, 1, 1

        layers += [ nn.Flatten(1) ]

        hiddens = [channels[-1] * h * w * d] + cfg.hiddens + [ cfg.outputs ]
        for i in range( len(hiddens)-1):
            if cfg.dropouts[i] > 0:
                layers += [ nn.Dropout(p=cfg.dropouts[i]) ]
            layers += [ nn.Linear(hiddens[i], hiddens[i+1] ) ]
            if i < len(hiddens)-2:
                layers += [ nn.ReLU() ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def default():
        return copy.deepcopy(Config(
            input     = None,       # input tensor shape:: (channels, depth, height, width)
            outputs   = None,       # output tensor shape;  sets in create()
            layers    = 2,          # layers in block
            channels  = [],         # conv3D chanels
            kernels   = 3,          # int or list: size of the convolutional kernel
            strides   = 1,          # int or list: stride of the convolutional kernel
            mode      = 'zeros',
            batchnorm = True,       # batchnorm
            bias      = False,      # int or list: padding around the image
            pools     = 0,          # pooling
            dropouts  = 0.,         # dropout rate
            hiddens   = [],         # add output hidden layers
            residual  = [],         # add residual path
            averpool  = True        # add average pooling
        ))

    @staticmethod
    def unit_test():
        cnn3D_cfg = Config(
                    input     = (1,256,256,256),
                    outputs   = 8,                   # output tensor shape;  sets in create()
                    kernels   = 3,                   # int or list: size of the convolutional kernel
                    channels  = [ 1,   4,   8],      # conv3D chanels
                    layers    = [ 1,   2,   2],      # layers in block
                    residual  = [ 1,   2,   2],      # add residual path
                    strides   = [ 1,   2,   1],      # int or list: stride of the convolutional kernel
                    pools     = [ 0,   2,   2],      # pooling
                    batchnorm = True,                # batchnorm
                    bias      = False,               # int or list: padding around the image
                    averpool  = True,                # add average pooling
                    hiddens   = [64],                # add output hidden layers
                    dropouts  = 0.15,                # dropout rate
                )

        model = CNN3D(cnn3D_cfg)
        x = torch.zeros((cnn3D_cfg.input), dtype=torch.float32).unsqueeze(0)
        y = model(x)
        res = y.shape[-1] == cnn3D_cfg.outputs
        print('unit test CNN3D:', y.shape[-1] == cnn3D_cfg.outputs)
        return res
