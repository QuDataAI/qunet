# QuNet - Models

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![PyPI version](https://badge.fury.io/py/torchinfo.svg)](https://badge.fury.io/py/torchinfo)

[QuNet](README.md)  gives access to various blocks with which you can build quite complex architectures.
<hr>

## MLP

Fully connected network with one or more hidden layers: `(B,*, input) -> (B,*, output)`.

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
```python
    mlp = MLP(input=32, stretch=4, output=1)
    y = mlp( torch.randn(1, 32) )
```
Can be created from config:
```python
    cfg = MLP.default()         
    cfg(input = 3, output = 1)  
    mlp = MLP(cfg)              
```
And also from the config and arguments:
```python
    mlp = MLP(cfg, hidden=[128, 512])
```

<hr>

## CNN

Simple convolutional network: `(B,C,H,W) ->  (B,C',H',W')`

The number of layers is set by the channel parameter. This is the number of channels at the output of each layer.
For example `input=(3,32,32)` and `channel=[8,16]` will create two CNN layers and the output of the module will be 16 channels.
The channel output size `(C',H',W')` is in cfg.output after the module is created.
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
```python
cfg = Config(input=(3,32,32), channel=[16,32])
cnn = CNN(cfg)
X = torch.empty( (1,) + cnn.cfg.input)
Y = cnn(X)
print(X.shape,"->",Y.shape)
```

<hr>

## SelfAttention

Self Attention

Args:         

* `emb:int`  - tokens embedding dim
* `n_heads:int` - number of heads emb % n_heads == 0 !            
* `drop=0` - dropout in attention and mlp            
* `res=1` - kind of skip-connections (0: none, 1: usial, 2: train one for all emb, 3: train for each emb)
* `casual=False` - kind of casual attention mask (True: GPT, False: Bert)
* `n_tokens=2048` -  maximum number of tokens (needed for causal==True)

<hr>

## TransformerBlock

One Transformer Block (it is all you need)
Args:         

* `emb:int` -  tokens embedding dim
* `n_heads:int` - number of heads emb % n_heads == 0 !            
* `drop=0` - dropout in attention and mlp
* `res=1` - kind of skip-connections (0: none, 1: usial, 2: train one for all emb, 3: train for each emb)
* `casual=False` - kind of casual attention mask (True: GPT, False: Bert)


<hr>

## Transformer

Transformer is all you need

Args:         

* `emb:int` - tokens embedding dim
* `n_heads` - number of heads emb % n_heads == 0 !
* `n_blocks=1` - number of transformer blocks
* `drop=0` - dropout in attention and mlp
* `res=1` - kind of skip-connections (0: none, 1: usial, 2: train one for all emb, 3: train for each emb)
* `casual=False` - kind of casual attention mask (True: GPT, False: Bert)

Example:
```python
trf = Transformer(n_blocks=5, emb=16, n_heads=2)

B, T, E = 1, 10, trf.cfg.block.att.emb
X = torch.empty( (B,T,E) )        
Y = trf(X)
print(X.shape,"->",Y.shape)
```

<hr>