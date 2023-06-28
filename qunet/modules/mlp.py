import copy
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn

from ..config import Config
from .total   import get_activation

#========================================================================================

class UnitTensor(nn.Module):
    def forward(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=-1)

#========================================================================================

class MLP(nn.Module):
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
            norm (bool = False):
                project a feature vector onto a sphere
            fun (str='gelu'):
                activation function: gelu, relu, sigmoid, tanh, relu6, swish, hswish, hsigmoid
            drop  (float:0.0):
                dropout at the output of the hidden layer

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
        super().__init__()
        self.cfg = MLP.default()
        self.cfg.set(*args, **kvargs)
        self.create()

        self.beta = 0.9    # степень усреднения (сглаживания) [0...1]
        self.datas_w = []  # усреднённые значения длин весов    weight
        self.datas_b = []  # усреднённые значения длин смещений bias
        self.grads_w = []  # усреднённые значения длин градиентов весов    weight
        self.grads_b = []  # усреднённые значения длин градиентов смещений bias

    #---------------------------------------------------------------------------

    @staticmethod
    def default():
        return copy.deepcopy(Config(
            input   = None,     # number of inputs > 0
            output  = None,     # number of outputs > 0
            hidden  = None,     # number of neurons in the hidden layer (int or list)
            stretch = None,     # if hiddem is None, then hidden = int(stretch*input)
            norm    = False,    # 
            fun     = 'gelu',   # activation function: gelu, relu, sigmoid, tanh
            drop    =  0,       # dropout at the output of the hidden layer
        ))

    #---------------------------------------------------------------------------

    def forward(self, x):
        x = self.layers(x)
        return x

    #---------------------------------------------------------------------------

    def prepare(self):
        cfg=self.cfg
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

    def create(self):
        self.prepare()
        seq = []
        for i in range (1, len(self.neurons)):
            seq += [ nn.Linear(self.neurons[i-1],  self.neurons[i]) ]
            if i+1 < len(self.neurons):
                seq += [get_activation(self.cfg.fun),
                        nn.Dropout(self.cfg.drop)     ]
                if self.cfg.norm:
                    seq += [ UnitTensor() ]
        self.layers = nn.Sequential(*seq)        

    #---------------------------------------------------------------------------

    def update(self):
        """
        Вызывается тренереном перед обнулением градиентов, если модуль добавлен fit(states=[])
        """
        i = 0
        for layer in self.layers:
            if type(layer) == nn.Linear:
                w = layer.weight.detach().square().mean().sqrt()
                if len(self.datas_w) == i: self.datas_w.append(w)                    
                else:
                    self.datas_w[i] = self.beta * self.datas_w[i]  + (1-self.beta) * w

                if layer.weight.grad is None: g = 0                    
                else:                         g = layer.weight.grad.square().mean().sqrt()
                if len(self.grads_w) == i: self.grads_w.append(g)                    
                else:
                    self.grads_w[i] = self.beta * self.grads_w[i]  + (1-self.beta) * g
                
                g, b = 0, 0
                if layer.bias is not None:
                    b = layer.bias.detach().square().mean().sqrt()

                    if layer.bias.grad is None: g = 0                    
                    else:                       g = layer.bias.grad.square().mean().sqrt()                    


                if len(self.datas_b) == i:  self.datas_b.append(b)                    
                else:
                    self.datas_b[i] = self.beta * self.datas_b[i]  + (1-self.beta) * b                

                if len(self.grads_b) == i:  self.grads_b.append(g)                    
                else:
                    self.grads_b[i] = self.beta * self.grads_b[i]  + (1-self.beta) * g
    
                i += 1
        
    #---------------------------------------------------------------------------

    def plot(self, w=12, h=3, eps=1e-8):
        fig, ax = plt.subplots(1,1, figsize=(w, h))        
        x = np.arange(len(self.datas_w))
        if self.datas_w:           
            ax.plot(x, np.array(self.datas_w).transpose(),  "-b.", lw=2,  label="weight")
            ax.set_ylim(bottom=0)   # after plot !!!            
            ax.set_ylabel("weight", color='b')
            ax.tick_params(axis='y', colors='b')
            ax.set_xticks(x)
            ax.grid(ls=":")

        if self.datas_b:           
            ax1 = ax.twinx()            
            ax1.plot(x, np.array(self.datas_b).transpose(), "-g.", label="bias")
            ax1.spines["left"].set_position(("outward", 40))
            ax1.spines["left"].set_visible(True)
            ax1.yaxis.set_label_position('left')
            ax1.yaxis.set_ticks_position('left')
            ax1.set_ylabel("bias", color='g')
            ax1.tick_params(axis='y', colors='g')

        if self.grads_w:
            ax2 = ax.twinx()            
            ax2.plot(x, np.array(self.grads_w).transpose(), "--b.", mfc='r', mec='r', label="grad")
            ax2.set_ylim(bottom=0)   # after plot !!!
            ax2.set_ylabel("--- grad weight",  color='r')
            ax2.tick_params(axis='y', colors='r')

        if self.grads_b:
            ax3 = ax.twinx()            
            ax3.plot(x, np.array(self.grads_b).transpose(), "--g.", mfc='r', mec='r', label="grad")
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

        mlp = MLP(cfg, hidden=128, norm=True)

        print("ok MLP")
        return True
    
#========================================================================================    
#                             MLP with skip connections                  
#========================================================================================

class ResBlockMLP(nn.Module):
    def __init__(self,  *args, **kvargs) -> None:
        super().__init__()
        self.cfg = ResBlockMLP.default()
        cfg = self.cfg.set(*args, **kvargs)
        
        assert cfg.mlp.input == cfg.mlp.output, f"In mlp should be input ({cfg.mlp.input}) == output ({cfg.mlp.output})"

        self.mlp  = MLP(cfg.mlp)
        self.norm = nn.LayerNorm(cfg.mlp.input)
        self.register_buffer("mult", torch.tensor(float(1.0)))
        self.register_buffer("std",  torch.tensor(float(0.0)))        

        self.debug  = False
        self.avr_x    = None
        self.avr_dx   = None
        self.grads    = []
        self.__hook   = None 

    #---------------------------------------------------------------------------

    @staticmethod
    def default():
        return copy.deepcopy(Config(
            mlp = MLP.default(),             
            beta  = 0.9,                        
        ))

    #---------------------------------------------------------------------------

    def forward(self, x):
        
        dx = self.mlp(self.norm(x))                # B,...,F

        if self.debug:                             # ema модуля параметров модели
            v_x  = torch.sqrt(torch.square(x. detach()).sum(-1).mean())
            v_dx = torch.sqrt(torch.square(dx.detach()).sum(-1).mean())
            b, b1 = self.cfg.beta, 1-self.cfg.beta
            if self.avr_x  is None: self.avr_x  = v_x
            else:                   self.avr_x  = b * self.avr_x  + b1 * v_x
            if self.avr_dx is None: self.avr_dx = v_dx            
            else:                   self.avr_dx = b * self.avr_dx + b1 * v_dx
        
        if self.training and self.std > 0:
            return self.mult * x  + dx * (1+torch.randn((x.size(0),1), device=x.device)*self.std)
        else:
            return self.mult * x  + dx
    
    #---------------------------------------------------------------------------

    def update(self):
        """
        Вызывается тренереном перед обнулением градиентов, если модуль добавлен fit(states=[])
        """

        i = 0
        for layer in self.mlp:
            if type(layer) == nn.Linear:
                if layer.weight.grad is None:
                    g = 0                    
                else:
                    g = layer.weight.grad.square().mean().sqrt()
                if len(self.grads) == i:
                    self.grads.append(g)                    
                else:
                    self.grads[i] = self.cfg.beta * self.grads[i]  + (1-self.cfg.beta) * g

                d = layer.weight.detach().square().mean().sqrt()
                if len(self.datas) == i:                    
                    self.datas.append(d)                    
                else:
                    self.datas[i] = self.beta * self.datas[i]  + (1-self.beta) * d

                i += 1


#========================================================================================    

class ResMLP(nn.Module):
    def __init__(self,  *args, **kvargs) -> None:
        """
        Args
        ------------
            n_blocks (int=2):
                number of transformer blocks
            mlp ( Config=MLP.default() ):
                should be: input == output

        Example
        ------------
        ```
            mlp = ResMLP(n_blocks=5, mlp=Config(input=32, hidden=[128,128], output=32))
            y = mlp( torch.randn(1, 32) )
        ```
        """
        super().__init__()
        self.cfg = ResMLP.default()
        cfg = self.cfg.set(*args, **kvargs)
        
        assert cfg.mlp.input == cfg.mlp.output, f"In mlp should be input ({cfg.mlp.input}) == output ({cfg.mlp.output})"

        self.blocks = nn.ModuleList([  ResBlockMLP(mlp=cfg.mlp) for _ in range(cfg.n_blocks) ]) 
        self.out = None
        if cfg.out is not None:
            self.out = MLP(cfg.out)
        self.mult(cfg.mults)   
        self.std(cfg.stds)       

    #---------------------------------------------------------------------------

    @staticmethod
    def default():
        return copy.deepcopy(Config(
            n_blocks = 1,
            mlp = MLP.default(),   
            out = None,          
            beta  = 0.9,
            mults = 1.0,     # can be list
            stds  = 0.0,     # can be list
        ))

    #---------------------------------------------------------------------------

    def forward(self, x):
        for block in self.blocks:            
            x = block(x)
        if self.out is not None:
            x = self.out(x)
        return x
    
    #---------------------------------------------------------------------------

    def debug(self, value=True, beta=None):
        for block in self.blocks:
            block.debug = value
            if not value:
                block.avr_x = block.avr_dx  = None        
                block.grads  = []

            if beta is not None:
                block.cfg.beta = beta

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
        if i is not None:
            assert type(stds) in [float, int] and i >=0 and i < self.cfg.n_blocks, f"Wrong stds={stds} for i={i}"    
            self.blocks[i].std.fill_(float(stds))
            return

        if type(stds) in [float, int]:
            stds = [stds] * self.cfg.n_blocks

        assert type(stds) in [list, tuple] and len(stds) == self.cfg.n_blocks, f"Wrong stds={stds}"
        for block, std in zip(self.blocks, stds):
            block.std.fill_(float(std))          

    #---------------------------------------------------------------------------

    def mult(self, mults, i=None):
        if i is not None:
            assert type(mults) in [float, int] and i >=0 and i < self.cfg.n_blocks, f"Wrong mults={mults} for i={i}"    
            self.blocks[i].mult.fill_(float(mults))
            return

        if type(mults) in [float, int]:
            mults = [mults] * self.cfg.n_blocks

        assert type(mults) in [list, tuple] and len(mults) == self.cfg.n_blocks, f"Wrong mults={mults}"
        for block, mult in zip(self.blocks, mults):
            block.mult.fill_(float(mult))

    #---------------------------------------------------------------------------

    def update(self):
        for block in self.blocks:
            block.update()    

    #---------------------------------------------------------------------------

    def plot(self, w=12,h=3,eps=1e-8):
        fig, ax = plt.subplots(1,1, figsize=(w, h))

        plt.text(0,0,f" mult\n std\n", ha='left', transform = ax.transAxes, fontsize=8)
        weights, dx = [], []
        for i,block in enumerate(self.blocks):
            ww = [d.cpu().item() for d in block.datas]
            weights.append(ww)
            ma = max(ma, len(ww))
            weights.append(ww)
            if block.avr_dx is not None:
                dx.append( (block.avr_dx/(block.avr_x+eps)).cpu().item() )

            plt.text(i,0,f"{block.mult.item():.2f}\n{block.std.item():.2f}\n", ha='center', fontsize=8)

        idxs = np.arange(self.cfg.n_blocks)        
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
        for block in self.blocks:                    
            grads.append([ g.cpu().item() for g in block.grads])
            ma = max(ma, len(block.grads))
        
        if len (grads):
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
    def unit_test():
        mlp = ResMLP(n_blocks=5, mlp=Config(input=32, hidden=[128,128], output=32))
        y = mlp( torch.randn(1, 32) )
        print(f"ok ResMLP: {y.shape}")
        return True



