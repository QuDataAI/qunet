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
