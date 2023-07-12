import copy
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn

from ..config import Config
from .total   import Create

#========================================================================================

class UnitTensor(nn.Module):
    def forward(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=-1)

#========================================================================================

class MLP(nn.Sequential):
    def __init__(self,  *args, **kvargs) -> None:
        """Fully connected network with one or more hidden layers:
        (B,*, input) -> (B,*, output).

        Args
        ------------
            input (int=None):
                number of inputs > 0
            output (int=None):
                number of outputs > 0
            hidden (int or list = None):
                number of neurons in the hidden layer
            stretch (int = None):            
                if there is, then hidden = int(stretch*input)
            norm (int = 0):
                kind of norm layer
            unit (bool = False):
                project a feature vector onto a sphere
            fun (str='gelu'):
                activation function: gelu, relu, sigmoid, tanh, relu6, swish, hswish, hsigmoid
            drop  (int = 0):
                kind of dropout (0 or 1) at the output of the hidden layer

        If there is more than one layer - hidden is a list of the number of neurons in each layer
        There may be no hidden layer: hidden == 0 or == [] or stretch == 0,
        then it's a normal input -> output line layer with no activation function

        Example
        ------------
        ```
            mlp = MLP(input=32, stretch=4, output=1)
            y = mlp( torch.randn(1, 32) )
        ```
        Can be created from config:
        ```
            cfg = MLP.default()
            cfg(input = 3, output = 1)
            mlp = MLP(cfg)
        ```
        And also from the config and key-val arguments:
        ```
            mlp = MLP(cfg, hidden=[128, 512])
        ```
        """        
        self.cfg = MLP.default()
        cfg = self.cfg.set(*args, **kvargs)
        layers = self.create(cfg)
        super().__init__(*layers)

        self.beta = 0.9    # степень усреднения (сглаживания) [0...1]
        self.datas_w = []  # усреднённые значения длин весов    weight
        self.datas_b = []  # усреднённые значения длин смещений bias
        self.grads_w = []  # усреднённые значения длин градиентов весов    weight
        self.grads_b = []  # усреднённые значения длин градиентов смещений bias
        self.data = torch.tensor(float(0))
        self.grad = torch.tensor(float(0))
    #---------------------------------------------------------------------------

    @staticmethod
    def default():
        return copy.deepcopy(Config(
            input   = None,     # number of inputs > 0
            output  = None,     # number of outputs > 0
            hidden  = None,     # number of neurons in the hidden layer (int or list)
            stretch = None,     # if hiddem is None, then hidden = int(stretch*input)
            norm    = 0,        # kind of norm layer
            unit    = False,    # project a feature vector onto a sphere
            fun     = 'gelu',   # activation function: gelu, relu, sigmoid, tanh            
            drop    =  0,       # kind of dropout (0 or 1) at the output of the hidden layer
        ))

    #---------------------------------------------------------------------------

    def prepare(self, cfg):        
        assert cfg.input is not None  and cfg.output is not None,  f'MLP: wrong input/output: {cfg.get_str()}'

        if type(cfg.hidden) is list:
            self.neurons = [cfg.input] + cfg.hidden + [cfg.output]
        else:
            if (cfg.hidden is None) and (cfg.stretch is not None):
                cfg.hidden = int(cfg.stretch * cfg.input)

            if cfg.hidden is None or cfg.hidden <= 0:
                self.neurons = [cfg.input, cfg.output]
            else:
                self.neurons = [cfg.input, cfg.hidden, cfg.output]

        if cfg.fun not in ['gelu', 'relu', 'sigmoid', 'tanh', 'relu6', 'swish', 'hswish', 'hsigmoid']:
            print(f"MLP warning: unknown activation function {cfg.fun}, set to gelu")
            cfg.fun  = 'gelu'

    #---------------------------------------------------------------------------

    def create(self, cfg):
        self.prepare(cfg)
        seq = []
        for i in range (1, len(self.neurons)):
            seq += [ nn.Linear(self.neurons[i-1],  self.neurons[i]) ]
            if i+1 < len(self.neurons):
                if cfg.norm:
                    seq += [ Create.norm(self.neurons[i], cfg.norm, 1)]

                seq += [Create.activation(cfg.fun)]

                if cfg.drop:
                    seq += [nn.Dropout(0.0)]
                if cfg.unit:
                    seq += [ UnitTensor() ]
        return seq

    #---------------------------------------------------------------------------

    def update(self):
        """
        Вызывается тренереном перед обнулением градиентов, если модуль добавлен fit(states=[])
        """
        i = 0
        for layer in self:
            if type(layer) == nn.Linear:
                w = layer.weight.detach().square().mean()
                if len(self.datas_w) == i: self.datas_w.append(w)                    
                else:
                    self.datas_w[i] = self.beta * self.datas_w[i]  + (1-self.beta) * w

                if layer.weight.grad is None: g = 0                    
                else:                         g = layer.weight.grad.square().mean()
                if len(self.grads_w) == i: self.grads_w.append(g)                    
                else:
                    self.grads_w[i] = self.beta * self.grads_w[i]  + (1-self.beta) * g
                
                g, b = 0, 0
                if layer.bias is not None:
                    b = layer.bias.detach().square().mean().sqrt()

                    if layer.bias.grad is None: g = 0                    
                    else:                       g = layer.bias.grad.square().mean()


                if len(self.datas_b) == i:  self.datas_b.append(b)                    
                else:
                    self.datas_b[i] = self.beta * self.datas_b[i]  + (1-self.beta) * b                

                if len(self.grads_b) == i:  self.grads_b.append(g)                    
                else:
                    self.grads_b[i] = self.beta * self.grads_b[i]  + (1-self.beta) * g
    
                i += 1
        if self.datas_w:
            self.data = sum(self.datas_w) / len(self.datas_w)
        if self.grads_w:
            self.grad = sum(self.grads_w) / len(self.grads_w)

    #---------------------------------------------------------------------------

    def decay(self):
        """        
        """        
        res = set()
        for layer in self.layers:
            if type(layer) == nn.Linear:
                res.add(layer.weight)
        return res
    
    #---------------------------------------------------------------------------

    def set_drop(self, value):    
        for layer in self.layers:
            if type(layer) == nn.Dropout:
                layer.p = value

    #---------------------------------------------------------------------------

    def plot(self, w=12, h=3, eps=1e-8):
        fig, ax = plt.subplots(1,1, figsize=(w, h))        
        x = np.arange(len(self.datas_w))
        if self.datas_w:           
            y = [d.sqrt().cpu().item() for d in self.datas_w]
            ax.plot(x, np.array(y).transpose(),  "-b.", lw=2,  label="weight")
            ax.set_ylim(bottom=0)   # after plot !!!            
            ax.set_ylabel("weight", color='b')
            ax.set_xlabel("layer")
            ax.tick_params(axis='y', colors='b')
            ax.set_xticks(x)
            ax.grid(ls=":")

        if self.datas_b:           
            ax1 = ax.twinx()    
            y = [d.sqrt().cpu().item() for d in self.datas_b]        
            ax1.plot(x, np.array(y).transpose(), "-g.", label="bias")
            ax1.spines["left"].set_position(("outward", 40))
            ax1.spines["left"].set_visible(True)
            ax1.yaxis.set_label_position('left')
            ax1.yaxis.set_ticks_position('left')
            ax1.set_ylabel("bias", color='g')
            ax1.tick_params(axis='y', colors='g')

        if self.grads_w:
            ax2 = ax.twinx()      
            ax2.set_yscale('log') 
            y = [ d.sqrt().cpu().item() for d in self.grads_w]      
            ax2.plot(x, np.array(y).transpose(), "--b.", mfc='r', mec='r', label="grad")            
            ax2.set_ylabel("--- grad weight",  color='r')
            ax2.tick_params(axis='y', colors='r')

        if self.grads_b:
            ax3 = ax.twinx()      
            ax3.set_yscale('log') 
            y = [d.sqrt().cpu().item() for d in self.grads_b]      
            ax3.plot(x, np.array(y).transpose(), "--g.", mfc='r', mec='r', label="grad")
            ax3.spines["right"].set_position(("outward", 50))
            ax3.set_ylabel("--- grad bias", color='r')
            ax3.tick_params(axis='y', colors='r')

        plt.show()

    #---------------------------------------------------------------------------

    @staticmethod
    def unit_test():
        mlp = MLP(input=32, stretch=4, output=1)
        y = mlp( torch.randn(1, 32) )

        cfg = MLP.default()
        cfg(input = 3, output = 1)
        mlp = MLP(cfg)

        mlp = MLP(cfg, hidden=[128, 512])

        mlp = MLP(cfg, hidden=128, unit=True)

        print("ok MLP")
        return True
    
